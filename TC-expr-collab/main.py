from tqdm import tqdm
from parse import parse_args
import torch
import os
from ogb.linkproppred import Evaluator

from utils import *
from model import GCN, MLP, SAGE
from dataprocess import load_data
from learn import train, eval, update_adj, eval_comprehensive
import math
import pickle as pkl

import wandb

def run(encoder, predictor, split_edge, data, optimizer, adj_list_dict, args, wandb, tcs):
    best_val_hit = -math.inf

    data.train_adj_aug = data.train_adj

    log_res = []

    if args.train:
        for epoch in range(1, 1 + args.epochs):
            loss = train(encoder, predictor, optimizer, data, split_edge['train']['edge'], args)

            if epoch % args.eval_steps == 0:
                ress = eval(encoder, predictor, data, evaluator, split_edge, args)

                # print('\n\n**********Evaluation Result@{}**********'.format(epoch))
                # for key, res in ress.items():
                #     print('**********{}**********'.format(key))
                #     print(res)

                if args.wandb and args.train:
                    wandb.log({'train_loss': loss,\
                               'train_acc': ress['train'][args.track_idx],\
                               'valid_acc': ress['valid'][args.track_idx],\
                               'test_acc': ress['test'][args.track_idx],\
                               'train-tc': data.train_tc,\
                               'valid-tc': data.valid_tc,\
                               'test-tc': data.test_tc,\
                               })

                    log_res.append([loss, ress['train'][args.track_idx], ress['valid'][args.track_idx], ress['test'][args.track_idx],\
                                    data.train_tc, data.valid_tc, data.test_tc])
                    

                    # print(log_res)


                if ress['valid'][args.track_idx] > best_val_hit:
                    best_val_hit = ress['valid'][args.track_idx]

                    ress_final = ress
                    
                    if args.save:
                        torch.save(encoder.state_dict(), os.path.join(args.path, 'model', args.dataset, args.model, 'encoder_{}.pt'.format(args.run)))
                        torch.save(predictor.state_dict(), os.path.join(args.path, 'model', args.dataset, args.model, 'predictor_{}.pt'.format(args.run)))
                        torch.save(data.train_adj_aug, os.path.join(args.path, 'model', args.dataset, args.model, 'train_adj_aug_{}.pt'.format(args.run)))

                        for key in tcs:
                            torch.save(tcs[key], os.path.join(args.path, 'model', args.dataset, args.model, '{}_tc_update_{}.pt'.format(key, args.run)))
            #50, 50
            if args.aug and epoch >= args.warm_up and epoch%args.update_interval == 0:
                data.train_adj_aug, tcs = update_adj(encoder, predictor, data, epoch, args)

                
    else:
        encoder.load_state_dict(torch.load(os.path.join(args.path, 'model', args.dataset, args.model, 'encoder_{}.pt'.format(args.run))))
        predictor.load_state_dict(torch.load(os.path.join(args.path, 'model', args.dataset, args.model, 'predictor_{}.pt'.format(args.run))))
        data.train_adj_aug = torch.load(os.path.join(args.path, 'model', args.dataset, args.model, 'train_adj_aug_{}.pt'.format(args.run)))

        if args.model == 'GCN-aug':
            data.train_adj_aug = torch.load(os.path.join(args.path, 'model', args.dataset, args.model, 'train_adj_aug_{}.pt'.format(args.run)))

        ress_final = eval_comprehensive(encoder, predictor, data, evaluator, split_edge, adj_list_dict, 'test', args)
    return ress_final, log_res




if __name__ == '__main__':
    args = parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()

    seed_everything(args.seed)
    path_everything(args.dataset, args.model)

    if args.dataset == 'collab':
        args.track_idx = 3 #H@50
    elif args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
        args.track_idx = 4 #H@100


    evaluator = Evaluator(name='ogbl-collab')
    train_hits, val_hits, test_hits = [], [], []
    pbar = tqdm(range(args.runs), unit='run')


    for args.run in pbar:
        seed_everything(args.seed + args.run)

        """build dataset"""
        data, split_edge, adj_list_dict, tcs = load_data(args)


        for key1 in split_edge:
            for key2 in split_edge[key1]:
                split_edge[key1][key2] = split_edge[key1][key2].to(args.device)

        data = data.to(args.device)

        if args.wandb and args.train:
            wand_setting = wandb.init(
                                project="TC-ICLR24-{}-new22".format(args.dataset),
                                notes="",
                                tags=["Baseline", args.model, "Run-{}".format(args.run)],\
                                name = args.model + '-' + args.dataset, 
                                config = args
                                )
            

        # print(args.encoder)
        """build encoder"""
        if args.encoder == 'GCN':
            encoder = GCN(data.x.shape[1], args.n_hidden, args.n_hidden, args.n_layers, args.en_dp).to(args.device)
        elif args.encoder == 'SAGE':
            encoder = SAGE(data.x.shape[1], args.n_hidden, args.n_hidden, args.n_layers, args.en_dp).to(args.device)
            

        """build link predictor"""
        if args.predictor == 'M-LP':
            predictor = MLP(args.n_hidden, args.n_hidden, 1, args.n_layers, args.lp_dp).to(args.device)

        if args.encoder in ['GCN', 'SAGE']:
            optimizer = torch.optim.Adam([{'params': encoder.parameters(), "lr": args.encoder_lr},
                                          {'params': predictor.parameters(), 'lr': args.predictor_lr}])

        ress_final, log_res = run(encoder, predictor, split_edge, data, optimizer, adj_list_dict, args, wandb, tcs)
        
        
        # print(ress_final['test'])
        train_hits.append(ress_final['train'])
        val_hits.append(ress_final['valid'])
        test_hits.append(ress_final['test'])

        if args.wandb and args.train:
            wandb.finish()

        pkl.dump(log_res, open(os.path.join(args.path, 'model', args.dataset, args.model, 'log_res_{}.pkl'.format(args.run)), 'wb'))

    print('Train_Hit:', np.mean(train_hits, axis = 0), np.std(train_hits, axis = 0))          
    print('Val_Hit:', np.mean(val_hits, axis = 0), np.std(val_hits, axis = 0))
    print('Test_Hit:', np.mean(test_hits, axis = 0), np.std(test_hits, axis = 0))



