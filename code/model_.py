import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import scipy


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        raise NotImplementedError

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),           
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, input_size),
        )

    def forward(self, x):
        out = self.linear(x)
        return out

class EGHG(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(EGHG, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        if self.config['mlp'] == 1:
            print('init mlp model')
            self.mlp = MLP(input_size=64)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def __init_weight(self): 
        self.num_users  = self.dataset.n_users      
        self.num_items  = self.dataset.m_items      
        self.latent_dim = self.config['latent_dim_rec']     
        self.n_layers = self.config['EGHG_n_layers']   
        self.keep_prob = self.config['keep_prob']      
        self.A_split = self.config['A_split']           
        self.embedding_user = torch.nn.Embedding(       
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(      
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:                
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            print('use xavier initilizer')
        self.f = nn.Sigmoid()  
        self.Graph = self.dataset.getSparseGraph() 
        print("using laplace k porcess")
        ones = scipy.sparse.eye(self.Graph.shape[0])
        ones= self._convert_sp_mat_to_sp_tensor(ones).to(world.device)
        graph_temp = (1-self.config['k']) * ones +self.config['k'] * self.Graph
        self.Graph = graph_temp.to(world.device)
        if self.config['cache'] == 1:
            print("doing cache tricks with beta = {} ".format(world.config['beta']))
            cache_emblist = []
        if self.config['mlp'] == 1:
            print("doing mlp tricks with alpha = {} ".format(world.config['alpha']))
            mlp_emblist = []
    def __dropout_x(self, x, keep_prob): 
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):    
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        if self.config['cache'] == 1:
            cache_emblist = []
        if self.config['mlp'] == 1:
            mlp_emblist = []  
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight 
        all_emb = torch.cat([users_emb, items_emb])  
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph 


        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    if self.config['mlp'] == 1:
                        mlp_emblist.append(all_emb)
                        mlp_emb = self.mlp(all_emb)
                        mlp_emblist.append(world.config['alpha'] * mlp_emb)
                        mlp_embs = torch.stack(mlp_emblist, dim=1)
                        all_emb = torch.mean(mlp_embs, dim=1)
                    if self.config['cache'] == 1:
                        cache_emblist.append(world.config['beta'] * all_emb)   # e0, e1, e2, e3
                        cache_embs = torch.stack(cache_emblist, dim=1)
                        cache_out = torch.mean(cache_embs, dim=1)
                        temp_emb.append(torch.sparse.mm(g_droped[f], cache_out))
                    else:
                        temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb

            else:
                if self.config['cache'] == 1:
                    cache_emblist.append(world.config['beta'] * all_emb)   # e0, e1, e2, e3
                    cache_embs = torch.stack(cache_emblist, dim=1)
                    cache_out = torch.mean(cache_embs, dim=1)
                    all_emb = torch.sparse.mm(g_droped, cache_out)
                if self.config['mlp'] == 1:
                    mlp_emblist.append(all_emb)
                    mlp_emb = self.mlp(all_emb)
                    mlp_emblist.append(world.config['alpha'] * mlp_emb)
                    mlp_embs = torch.stack(mlp_emblist, dim=1)
                    all_emb = torch.mean(mlp_embs, dim=1)
                else:
                    all_emb = torch.sparse.mm(g_droped, all_emb)

            embs.append(all_emb) 
        embs = torch.stack(embs, dim=1) 
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long()) 
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
