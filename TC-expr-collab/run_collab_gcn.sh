python main.py --dataset='collab' --n_layers=3 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=2 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN' --save --remove_rep --train --wandb

python main.py --dataset='collab' --n_layers=3 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=2 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN-aug' --save --remove_rep --aug --gcn_denoise --train --wandb
