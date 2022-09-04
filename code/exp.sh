nohup python -u main.py --decay=1e-4 --lr=0.003 --layer=6 --dataset="AMusic" --GPU 2 --cache 1 --Hadj 1 --epochs 5000 --dropadj 0.5 --load_adj H_adj --k 1 > log/AMusic/AMusic_EVA.log 2>&1 &
nohup python -u main.py --decay=1e-4 --lr=0.002 --layer=12 --dataset="lastfm" --GPU 4 --cache 1 --Hadj 1 --epochs 5000 --dropadj 0.3 --load_adj H_adj --k 1 > log/lastfm_EVA.log 2>&1 &
nohup python -u main.py --decay=1e-4 --lr=0.002 --layer=6 --dataset="ml-1m" --GPU 4 --cache 0 --Hadj 1 --epochs 3000 --dropadj 0.7 --load_adj H_adj --Enhanced=0 --k_G=1 --k_HG=1 > log/ml-1m/1_1_0.log 2>&1 &
nohup python -u main.py --decay=1e-4 --lr=0.005 --layer=8 --dataset="AToy" --GPU 0 --cache 1 --Hadj 1 --epochs 5000 --dropadj 0.3 --load_adj H_adj --Enhanced=0  --k_G=1 --k_HG=1 > log/AToy/1_1_0.log 2>&1 &