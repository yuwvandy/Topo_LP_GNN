import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", nargs="?", default="Cora")

    # model
    parser.add_argument("--encoder", type=str, default="GCN")
    parser.add_argument("--predictor", type=str, default="M-LP")
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=256)

    # training
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--train_bsz', type=int, default=64*1024)
    parser.add_argument('--eval_bsz', type=int, default=4096)
    parser.add_argument('--eval_node_bsz', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--encoder_lr', type=float, default=1e-3)
    parser.add_argument('--predictor_lr', type=float, default=1e-3)
    parser.add_argument('--en_dp', type=float, default=0.0)
    parser.add_argument('--lp_dp', type=float, default=0.0)

    # specific for LP
    parser.add_argument('--topks', default=[5, 10, 20, 50, 100])

    # experiments
    parser.add_argument("--seed", type=int, default=1028,
                        help="seed to run the experiment")
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument('--model', type=str, default='NCN')

    parser.add_argument("--save", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--aug", action='store_true')
    parser.add_argument("--wandb", action='store_true')

    parser.add_argument("--warm_up", type = int, default = 50)
    parser.add_argument("--update_interval", type = int, default = 50)
    parser.add_argument('--alpha', type=float, default=0.1)


    parser.add_argument('--use_val', action='store_true')
    parser.add_argument('--mask_edge_in_prop', action='store_true')
    parser.add_argument('--sage_norm', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--ln', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--aug_type', type=str, default='gcn')   




    return parser.parse_args()
