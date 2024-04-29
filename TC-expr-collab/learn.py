from utils import *
import torch
import os
from torch_sparse import SparseTensor
import torch_sparse
from torch_geometric.utils import softmax, negative_sampling
import torch.nn.functional as F
from dataprocess import PermIterator, update_tc
from scipy.stats import rankdata
from itertools import product
from torch_scatter import segment_csr


def train(encoder, predictor, optimizer, data, train_edge, args):
    encoder.train()
    predictor.train()

    neg_edge = negative_sampling(data.train_edge_index, num_neg_samples = train_edge.shape[0]).t()

    total_loss, count = 0, 0
    for batch in PermIterator(train_edge.device, train_edge.shape[0], args.train_bsz):
        h = encoder(data.x, data.train_adj_aug)

        pos_score = predictor(h[train_edge[batch, 0]]*h[train_edge[batch, 1]])
        pos_loss = -F.logsigmoid(pos_score).mean()

        neg_score = predictor(h[neg_edge[batch, 0]]*h[neg_edge[batch, 1]])
        neg_loss = -F.logsigmoid(-neg_score).mean()

        loss = (pos_loss + neg_loss) / 2

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item() * train_edge.shape[0]
        count += train_edge.shape[0]

    return total_loss / count


@torch.no_grad()
def eval(encoder, predictor, data, evaluator, split_edge, args):
    encoder.eval()
    predictor.eval()

    ress = {'train': [],
            'valid': [],
            'test': []}

    h = encoder(data.x, data.train_adj_aug)

    # eval_per_edge
    for key in split_edge:
        # if key == 'test' and args.dataset == 'collab':
        #     h = encoder(data.x, data.train_val_adj)

        edge, neg_edge = split_edge[key]['edge'], split_edge[key]['edge_neg']

        pos_preds = torch.cat([predictor(h[edge[batch, 0]]*h[edge[batch, 1]]).squeeze().cpu() \
            for batch in PermIterator(edge.device, edge.shape[0], args.eval_bsz, training = False)])
        neg_preds = torch.cat([predictor(h[neg_edge[batch, 0]]*h[neg_edge[batch, 1]]).squeeze().cpu() \
            for batch in PermIterator(neg_edge.device, neg_edge.shape[0], args.eval_bsz, training = False)])


        for K in args.topks:
            evaluator.K = K
            hits = evaluator.eval({'y_pred_pos': pos_preds, 'y_pred_neg': neg_preds})[f'hits@{K}']

            ress[key].append(hits)

    return ress




@torch.no_grad()
def update_adj(encoder, predictor, data, epoch, args):
    encoder.eval()
    predictor.eval()

    h_train = encoder(data.x, data.train_adj_aug)

    if args.gcn_denoise:
        h_node = data.train_adj_gcn.matmul(h_train)[data.train_edge_index[1]]
    else:
        h_node = data.train_adj_sage.matmul(h_train)[data.train_edge_index[1]]

    h_pos_node = h_train[data.train_edge_index[0]]

    score = predictor.score(h_node*h_pos_node).squeeze()
    score = softmax(score, ptr = data.train_ptr, dim = 0, num_nodes = data.x.shape[0])

    norm_origin = segment_csr(data.train_edge_weight, data.train_ptr, reduce='sum')[data.train_edge_index[1]]

    weight = (score * norm_origin).detach()

    new_weight = args.alpha*weight + data.train_edge_weight

    adj_t = SparseTensor(
        row=data.train_edge_index[0], col=data.train_edge_index[1], value=new_weight, is_sorted=False)
    
    train_tc, valid_tc, test_tc = update_tc(data, new_weight)

    torch.save(train_tc, os.path.join(args.path, 'model', args.dataset, args.model, 'train_tc_update.pt'))
    torch.save(valid_tc, os.path.join(args.path, 'model', args.dataset, args.model, 'valid_tc_update.pt'))
    torch.save(test_tc, os.path.join(args.path, 'model', args.dataset, args.model, 'test_tc_update.pt'))
    
    data.train_tc = train_tc[(data.deg['train'] != 0) & (data.one_hot_dict['test'])].mean().item()
    data.valid_tc = valid_tc[(data.deg['valid'] != 0) & (data.one_hot_dict['test'])].mean().item()
    data.test_tc = test_tc[(data.deg['test'] != 0) & (data.one_hot_dict['test'])].mean().item()


    return adj_t, {'train': train_tc.cpu(), 'valid': valid_tc.cpu(), 'test': test_tc.cpu()}




@torch.no_grad()
def eval_comprehensive(encoder, predictor, data, evaluator, split_edge, adj_list_dict, eval_type, args):
    encoder.eval()
    predictor.eval()

    ress = {'train': [],
            'valid': [],
            'test': []}

    h = encoder(data.x, data.train_adj_aug)

    #Eval edge
    for key in split_edge:
        edge, neg_edge = split_edge[key]['edge'], split_edge[key]['edge_neg']

        pos_preds = torch.cat([predictor(h[edge[batch, 0]]*h[edge[batch, 1]]).squeeze().cpu() \
            for batch in PermIterator(edge.device, edge.shape[0], args.eval_bsz, training = False)])
        neg_preds = torch.cat([predictor(h[neg_edge[batch, 0]]*h[neg_edge[batch, 1]]).squeeze().cpu() \
            for batch in PermIterator(neg_edge.device, neg_edge.shape[0], args.eval_bsz, training = False)])


        for K in args.topks:
            evaluator.K = K
            hits = evaluator.eval({'y_pred_pos': pos_preds, 'y_pred_neg': neg_preds})[f'hits@{K}']

            ress[key].append(hits)

    #Eval node
    ratings_list = []
    groundTruth_nodes_list = []
    nodes_list = []
    mrr_list= []

    node = data.eval_node[eval_type]
    print(node.shape[0] // args.eval_node_bsz)

    for count, batch in enumerate(PermIterator(data.x.device, node.shape[0], args.eval_node_bsz, training = False)):
        print(count)
        batch_node = node[batch]

        score = (h[batch_node].unsqueeze(1) * h.unsqueeze(0)).view(batch.shape[0]*h.shape[0], -1)

        score = predictor(score).detach().cpu().squeeze().view(batch_node.shape[0], h.shape[0])

        if eval_type == 'train':
            clicked_nodes = [np.array([])]
            groundTruth_nodes = [list(adj_list_dict['train'][node.item()]) for node in batch_node]

        elif eval_type == 'valid':
            clicked_nodes = [np.array(list(adj_list_dict['train'][node.item()]), dtype = int) for node in batch_node]
            groundTruth_nodes = [list(adj_list_dict['valid'][node.item()]) for node in batch_node]

        elif eval_type == 'test':
            clicked_nodes = [np.array((list(adj_list_dict['train'][node.item()]) + list(adj_list_dict['valid'][node.item()]))) for node in batch_node]
            groundTruth_nodes = [list(adj_list_dict['test'][node.item()]) for node in batch_node]

        exclude_index, exclude_nodes = [], []
        for range_i, nodes in enumerate(clicked_nodes):
            exclude_index.extend([range_i] * len(nodes))
            exclude_nodes.extend(nodes)

        if args.dataset not in ['collab']:
            score[exclude_index, exclude_nodes] = -(1 << 10)
        rating_K = torch.topk(score, k = max(args.topks))[1]


        rating_ranking = rankdata(-np.array(score), method = 'ordinal', axis = 1)
        for i in range(len(groundTruth_nodes)):
            rank = min(rating_ranking[i][list(groundTruth_nodes[i])])
            mrr_list.append(1/rank)

        ratings_list.append(rating_K)
        groundTruth_nodes_list.append(groundTruth_nodes)
        nodes_list.append(batch_node.tolist())

    recall_list, ndcg_list, hit_ratio_list, precision_list, F1_list = [], [], [], [], []

    for nodes, X in zip(nodes_list, zip(ratings_list, groundTruth_nodes_list)):
        recalls, ndcgs, hit_ratios, precisions, F1s = test_one_batch_group(X, args.topks)

        recall_list.append(recalls)
        ndcg_list.append(ndcgs)
        hit_ratio_list.append(hit_ratios)
        precision_list.append(precisions)
        F1_list.append(F1s)

    recall_list = np.concatenate(recall_list)
    ndcg_list = np.concatenate(ndcg_list)
    hit_ratio_list = np.concatenate(hit_ratio_list)
    precision_list = np.concatenate(precision_list)
    F1_list = np.concatenate(F1_list)
    mrr_list = np.array(mrr_list)

    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model + '/' + str(args.run) + '_recall_list_' + eval_type + '.npy', recall_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model + '/' + str(args.run) + '_ndcg_list_' + eval_type + '.npy', ndcg_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model + '/' + str(args.run) + '_hit_ratio_list_' + eval_type + '.npy', hit_ratio_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model + '/' + str(args.run) + '_precision_list_' + eval_type + '.npy', precision_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model + '/' + str(args.run) + '_F1_list_' + eval_type + '.npy', F1_list)
    np.save(os.getcwd() + '/res/' + args.dataset + '/' + args.model + '/' + str(args.run) + '_mrr_list_' + eval_type + '.npy', mrr_list)

    return ress







# @torch.no_grad()
# def update_adj_aug(encoder, predictor, data):
#     encoder.eval()
#     predictor.eval()

#     h_train = encoder(data.x, data.train_val_adj)

#     h_node = data.train_val_adj.matmul(h_train)[data.train_val_edge_index[1]]

#     h_pos_node = h_train[data.train_val_edge_index[0]]

#     score = predictor.score(h_node*h_pos_node).squeeze()
#     score = softmax(score, ptr = data.train_val_ptr, dim = 0, num_nodes = data.x.shape[0])

#     norm_origin = segment_csr(data.train_val_edge_weight, data.train_val_ptr, reduce='sum')[data.train_val_edge_index[1]]

#     weight = (score * norm_origin).detach()

#     adj_t = SparseTensor(
#         row=data.train_val_edge_index[0], col=data.train_val_edge_index[1], value=1 * weight + 1 * data.train_val_edge_weight, is_sorted=False)

#     return adj_t
