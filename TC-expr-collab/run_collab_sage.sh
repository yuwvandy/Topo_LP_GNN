python main.py --dataset='collab' --n_layers=3 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=2 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE' --save --gcn_denoise --remove_rep --train --wandb

python main.py --dataset='collab' --n_layers=3 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=2 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE-aug' --save --aug --gcn_denoise --remove_rep --train --wandb

