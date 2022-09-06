# Effective Hybrid Graph and Hypergraph Convolution Network for Collaborative Filtering



## EHGCN-pytorch

This is the Pytorch implementation for our paper:

> Li, X., Guo, R., Chen, J. *et al.* Effective hybrid graph and hypergraph convolution network for collaborative filtering. *Neural Comput & Applic* (2022).[Paper](https://link.springer.com/article/10.1007/s00521-022-07735-y)




## Dataset

LastFM dataset is available at https://github.com/gusye1234/LightGCN-PyTorch. AMusic, AToy, and ML-1M datasets are available at https://github.com/familyld/DeepCF.




## An example to run a 6-layer EHGCN

run EHGCN on **AMusic** dataset:

```bash
cd code && python main.py --decay=1e-4 --lr=0.003 --layer=6 --dataset="AMusic" --GPU 2 --cache 1 --Hadj 1 --epochs 5000 --dropadj 0.5 --load_adj H_adj --k_G=1 --k_HG=1
```



## Results

*all metrics is under top-20*

- **ML-1M**

|              | Recall     | ndcg       |
| :----------: | ---------- | ---------- |
| **layer=2**  | 0.2903     | 0.2985     |
| **layer=4**  | 0.2998     | 0.3085     |
| **layer=6**  | **0.3041** | **0.3125** |
| **layer=8**  | 0.2995     | 0.3088     |
| **layer=10** | 0.2964     | 0.3052     |

- **AToy**

|              | Recall     | ndcg       |
| :----------: | ---------- | ---------- |
| **layer=2**  | 0.227      | 0.0104     |
| **layer=4**  | 0.0234     | 0.0102     |
| **layer=6**  | 0.0235     | 0.0104     |
| **layer=8**  | **0.0243** | **0.0109** |
| **layer=10** | 0.0240     | 0.0107     |

