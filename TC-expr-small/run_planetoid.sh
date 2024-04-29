python main.py --dataset='Cora' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN' --train_bsz=1152 --eval_bsz=8192 --train
python main.py --dataset='Cora' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN-aug' --train_bsz=1152 --eval_bsz=8192 --train --aug --alpha=2 --update_interval=1 --warm_up=1 --aug_type='gcn'
python main.py --dataset='Cora' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN-aug' --train_bsz=1152 --eval_bsz=8192 --train --aug --alpha=2 --update_interval=1 --warm_up=1 --aug_type='sage'

python main.py --dataset='Cora' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE' --train_bsz=1152 --eval_bsz=8192 --train --sage_norm
python main.py --dataset='Cora' --n_layers=1 --n_hidden=128 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE-aug' --train_bsz=1152 --eval_bsz=8192 --train --aug --alpha=0.01 --update_interval=1 --warm_up=10 --sage_norm --aug_type='gcn'
python main.py --dataset='Cora' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE-aug' --train_bsz=1152 --eval_bsz=8192 --train --aug --alpha=0.2 --update_interval=2 --warm_up=2 --sage_norm --aug_type='sage'


python main.py --dataset='Citeseer' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN' --train_bsz=1152 --eval_bsz=8192 --train
python main.py --dataset='Citeseer' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN-aug' --train_bsz=1152 --eval_bsz=8192 --train --aug --alpha=2 --update_interval=1 --warm_up=5 --aug_type='gcn'
python main.py --dataset='Citeseer' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN-aug' --train_bsz=1152 --eval_bsz=8192 --train --aug --alpha=2 --update_interval=1 --warm_up=5 --aug_type='sage'

python main.py --dataset='Citeseer' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE' --train_bsz=1152 --eval_bsz=8192 --train
python main.py --dataset='Citeseer' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE-aug' --train_bsz=1152 --eval_bsz=8192 --train --aug --alpha=2 --update_interval=1 --warm_up=1 --aug_type='gcn'
python main.py --dataset='Citeseer' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE-aug' --train_bsz=1152 --eval_bsz=8192 --train --aug --alpha=2 --update_interval=1 --warm_up=1 --aug_type='sage'



python main.py --dataset='Pubmed' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN' --train_bsz=1152 --eval_bsz=8192 --train
python main.py --dataset='Pubmed' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN-aug' --train_bsz=1152 --eval_bsz=8192 --train --aug --alpha=0.2 --update_interval=2 --warm_up=5 --aug_type='gcn'
python main.py --dataset='Pubmed' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN-aug' --train_bsz=1152 --eval_bsz=8192 --train --aug --alpha=0.5 --update_interval=2 --warm_up=5 --aug_type='sage'


python main.py --dataset='Pubmed' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE' --train_bsz=1152 --eval_bsz=8192 --train
python main.py --dataset='Pubmed' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE-aug' --train_bsz=1152 --eval_bsz=8192 --train --aug --alpha=0.5 --update_interval=2 --warm_up=5 --aug_type='gcn'
python main.py --dataset='Pubmed' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE-aug' --train_bsz=1152 --eval_bsz=8192 --train --aug --alpha=0.5 --update_interval=2 --warm_up=5 --aug_type='sage'


python main.py --dataset='Reptile' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN' --train_bsz=256 --eval_bsz=256 --train
python main.py --dataset='Reptile' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN-aug' --train_bsz=256 --eval_bsz=256 --train --aug --alpha=3 --update_interval=20 --warm_up=50 --aug_type='gcn'
python main.py --dataset='Reptile' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN-aug' --train_bsz=256 --eval_bsz=256 --train --aug --alpha=3 --update_interval=20 --warm_up=50 --aug_type='sage'

python main.py --dataset='Reptile' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE' --train_bsz=256 --eval_bsz=256 --train --sage_norm
python main.py --dataset='Reptile' --n_layers=1 --n_hidden=64 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE-aug' --train_bsz=256 --eval_bsz=256 --train --aug --alpha=1 --update_interval=1 --warm_up=1 --sage_norm --aug_type='gcn'
python main.py --dataset='Reptile' --n_layers=1 --n_hidden=64 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE-aug' --train_bsz=256 --eval_bsz=256 --train --aug --alpha=1 --update_interval=1 --warm_up=1 --sage_norm --aug_type='sage'



python main.py --dataset='Vole' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN' --train_bsz=256 --eval_bsz=256 --train
python main.py --dataset='Vole' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='GCN' --predictor='M-LP' --epochs=1000 --model='GCN-aug' --train_bsz=256 --eval_bsz=256 --train --aug --alpha=1 --update_interval=30 --warm_up=40 --aug_type='gcn'

python main.py --dataset='Vole' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE' --train_bsz=256 --eval_bsz=256 --train --sage_norm
python main.py --dataset='Vole' --n_layers=1 --n_hidden=256 --encoder_lr=0.001 --predictor_lr=0.001 --runs=10 --encoder='SAGE' --predictor='M-LP' --epochs=1000 --model='SAGE-aug' --train_bsz=256 --eval_bsz=256 --train --aug --alpha=0.5 --update_interval=20 --warm_up=40 --sage_norm --aug_type='gcn'



