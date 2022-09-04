import world
import torch
from dataloader import BasicDataset
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np
import scipy
import math


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


class EGHG(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(EGHG, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['EGHG_n_layers']
        self.keep_prob = self.config['keep_prob']
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
        self.Graph = self.Graph.cpu().to_dense()
        ## HG Laplacian
        ones = scipy.sparse.eye(self.dataset.n_users)
        ones = self._convert_sp_mat_to_sp_tensor(ones)
        HG_temp1 = (1 - self.config['k_HG']) * ones.to_dense() + self.config['k_HG'] * self.Graph[:self.dataset.n_users,
                                                                                       :self.dataset.n_users]
        ones = scipy.sparse.eye(self.dataset.m_items)
        ones = self._convert_sp_mat_to_sp_tensor(ones)
        HG_temp2 = (1 - self.config['k_HG']) * ones.to_dense() + self.config['k_HG'] * self.Graph[self.dataset.n_users:,
                                                                                       self.dataset.n_users:]

        ## G Laplacian

        self.Graph[:self.dataset.n_users, :self.dataset.n_users] = 0
        self.Graph[self.dataset.n_users:, self.dataset.n_users:] = 0
        ones = scipy.sparse.eye(self.Graph.shape[0])
        ones = self._convert_sp_mat_to_sp_tensor(ones)
        self.Graph = (1 - self.config['k_G']) * ones.to_dense() + self.config['k_G'] * self.Graph

        self.Graph[:self.dataset.n_users, :self.dataset.n_users] = HG_temp1
        self.Graph[self.dataset.n_users:, self.dataset.n_users:] = HG_temp2
        self.Graph = self.Graph.to(world.device)
        self.tOrder = 1
        if self.config['Enhanced'] != 0:
            print("using Enhanced convolution kerenl")
            self.Graph = torch.pow(self.Graph, 2)
        if self.config['useT'] != 1:
            print("using t-order laplacian convolution kernel")
            while (self.tOrder != self.config['useT']):
                self.Graph = torch.sparse.mm(self.Graph, self.Graph.to_dense())
                self.tOrder = self.tOrder + 1
            print("t-order process done!")
        if self.config['useW'] == 1:
            self.weight = Parameter(torch.FloatTensor(self.latent_dim, self.latent_dim))
            self.bias = Parameter(torch.FloatTensor(self.latent_dim))
            self.reset_parameters()
        if self.config['useA'] == 1:
            self.graphActive = nn.functional.relu
        if self.config['cache'] == 1:
            print("======   doing cache tricks   ======")
            cache_emblist = []

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def computer(self):
        if self.config['cache'] == 1:
            cache_emblist = []
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.config['cache'] == 1:
                cache_emblist.append(all_emb)  # e0, e1, e2, e3
                cache_embs = torch.stack(cache_emblist, dim=1)
                cache_out = torch.mean(cache_embs, dim=1)
                if self.config['useW'] == 1:
                    cache_out = torch.mm(cache_out, self.weight)
                #
                all_emb = torch.sparse.mm(g_droped, cache_out)
                #
                if self.config['useW'] == 1:
                    all_emb = all_emb + self.bias
                if self.config['useA'] == 1:
                    all_emb = self.graphActive(all_emb)
            if self.config['useW'] == 1:
                all_emb = torch.mm(all_emb, self.weight)
            #
            all_emb = torch.sparse.mm(g_droped, all_emb)
            #
            if self.config['useW'] == 1:
                all_emb = all_emb + self.bias
            if self.config['useA'] == 1:
                all_emb = self.graphActive(all_emb)

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
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
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
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
