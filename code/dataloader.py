import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
from myUtils import *

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        raise NotImplementedError
    
    def getSparseGraph(self):
        raise NotImplementedError

class LastFM(BasicDataset):
    def __init__(self, path="../data/lastfm"):
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        self.path = path
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None and world.config['load_adj'] == 'simple_adj':
            print('------------------------load simple_adj------------------------')
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            print(type(dense))
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
            return self.Graph
        if self.Graph is None and world.config['load_adj'] == 'H_adj':
            try:
                if world.config['Hadj'] == 1:
                    self.Graph = torch.load(self.path + '/s_pre_adj_mat_HyperGraph1_keepprob{}.pt'.format(world.config['dropadj']))
                    self.Graph = self.Graph.coalesce().to(world.device)
                    print('load H_adj in one-hop done')           
                    return self.Graph
                if world.config['Hadj'] == 2:
                    self.Graph = torch.load(self.path + '/s_pre_adj_mat_HyperGraph2_keepprob{}.pt'.format(world.config['dropadj']))
                    self.Graph = self.Graph.coalesce().to(world.device)
                    print('load H_adj in two-hop done')
                    return self.Graph
            except : 
                print('------------------------generate H_adj------------------------')

                if world.config['Hadj'] == 1:
                    print("generate Hadj in one-hop")
                    user_dim = torch.LongTensor(self.trainUser)                    
                    item_dim = torch.LongTensor(self.trainItem)                    
                    first_sub = torch.stack([user_dim, item_dim + self.n_users])                                           
                    second_sub = torch.stack([item_dim+self.n_users, user_dim])    
                    index = torch.cat([first_sub, second_sub], dim=1)              
                    data = torch.ones(index.size(-1)).int()                      


                    self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
                    dense = self.Graph.to_dense()
                    D = torch.sum(dense, dim=1).float()
                    D[D==0.] = 1.
                    D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
                    dense = dense/D_sqrt
                    dense = dense/D_sqrt.t()
                    index = dense.nonzero()
                    data  = dense[dense >= 1e-9]
                    assert len(index) == len(data)
                    self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
                    numpy_Graph = self.Graph.to_dense().numpy()
                    temp_Graph = sp.csr_matrix(numpy_Graph).tolil()
                    R = temp_Graph[:self.n_users, self.n_users:]
                    print("R", R.shape)
                    H_u = R
                    H_i = R.transpose()
                    D_u_v = H_u.sum(axis=1).reshape(1, -1)
                    D_u_e = H_u.sum(axis=0).reshape(1, -1)
                    arr1, arr2 = np.where(D_u_v==0)
                    for i in range(len(arr1)):
                        D_u_v[arr1[i], arr2[i]] = 1
                    arr1, arr2 = np.where(D_u_e==0)
                    for i in range(len(arr1)):
                        D_u_e[arr1[i], arr2[i]] = 1
                    temp1 = (H_u.transpose().multiply(np.sqrt(1.0/D_u_v))).transpose()      # D_uv(-1/2) * H_u
                    temp2 = temp1.transpose()                                               #        
                    A_u = temp1.multiply(1.0/D_u_e).dot(temp2)                              # D_uv(-1/2) * H_u * D_ue(-1) * H_uT * D_uv(-1/2)
                                                                    
                    print('A_u', A_u.get_shape())
                    D_i_v = H_i.sum(axis=1).reshape(1, -1)
                    D_i_e = H_i.sum(axis=0).reshape(1, -1)
                    arr1, arr2 = np.where(D_i_v==0)
                    for i in range(len(arr1)):
                        D_i_v[arr1[i], arr2[i]] = 1
                    arr1, arr2 = np.where(D_i_e==0)
                    for i in range(len(arr1)):
                        D_i_e[arr1[i], arr2[i]] = 1
                    temp1 = (H_i.transpose().multiply(np.sqrt(1.0 / D_i_v))).transpose()    
                    temp2 = temp1.transpose()                                              
                    A_i = temp1.multiply(1.0 / D_i_e).dot(temp2)                          
                                                                                    
                    print('A_i', A_i.get_shape())
                    if world.config['dropadj'] != 1:
                        A_u = drop_scipy_matrix(A_u, keep_prob=world.config['dropadj'])
                        A_i = drop_scipy_matrix(A_i, keep_prob=world.config['dropadj'])
                    temp_Graph[:self.n_users, :self.n_users] = A_u
                    temp_Graph[self.n_users:, self.n_users:] = A_i  
                    temp_Graph = temp_Graph.tocoo()
                    values = temp_Graph.data
                    indices = np.vstack((temp_Graph.row, temp_Graph.col))
                    i = torch.LongTensor(indices)
                    v = torch.FloatTensor(values)
                    shape = temp_Graph.shape

                    self.Graph = torch.sparse.FloatTensor(i, v, torch.Size(shape))
                    torch.save(self.Graph, self.path + '/s_pre_adj_mat_HyperGraph1_keepprob{}.pt'.format(world.config['dropadj']))
                    print('save norm_adj in' + self.path + '/s_pre_adj_mat_HyperGraph1_keepprob{}.pt'.format(world.config['dropadj']))
                    self.Graph = self.Graph.coalesce().to(world.device)
                    return self.Graph
                if world.config['Hadj'] == 2:
                ## hypergraph 2
                    user_dim = torch.LongTensor(self.trainUser)                     
                    item_dim = torch.LongTensor(self.trainItem)                     
                    first_sub = torch.stack([user_dim, item_dim + self.n_users])    
                    second_sub = torch.stack([item_dim+self.n_users, user_dim])     
                    index = torch.cat([first_sub, second_sub], dim=1)               
                    data = torch.ones(index.size(-1)).int()                         
                    self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
                    dense = self.Graph.to_dense()
                    D = torch.sum(dense, dim=1).float()
                    D[D==0.] = 1.
                    D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
                    dense = dense/D_sqrt
                    dense = dense/D_sqrt.t()
                    index = dense.nonzero()
                    data  = dense[dense >= 1e-9]
                    assert len(index) == len(data)
                    print("generate Hadj in two-hop")
                    self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
                    numpy_Graph = self.Graph.to_dense().numpy()
                    temp_Graph = sp.csr_matrix(numpy_Graph).tolil()
                    R = temp_Graph[:self.n_users, self.n_users:]

                    Hyper_part1 = R.transpose().dot(R)                                  # H_u = H || ( H ( HT H ) )
                    Hyper_temp1 = R.dot(Hyper_part1)                                    # ( H ( HT H ) )
                    Hyper_part2 = R.dot(R.transpose())                                  # H_i = HT || ( HT ( H HT ) )
                    Hyper_temp2 = R.transpose().dot(Hyper_part2)                        # ( HT ( H HT ) ) 

                    Hyper_temp1 = Hyper_temp1.A
                    Hyper_temp2 = Hyper_temp2.A

                    R = R.A

                    H_u = np.concatenate((R, Hyper_temp1), axis=1)                      # H_u = u,i || Hyper_temp1 -> u, 2i
                    H_i = np.concatenate((R.transpose(), Hyper_temp2), axis=1)          # h_I = i, u || Hyper_temp2 -> i, 2u
                    H_u = sp.lil_matrix(H_u)
                    H_i = sp.lil_matrix(H_i)
                    print("H_u", type(H_u), H_u.get_shape())
                    print("H_i", type(H_i), H_i.get_shape())
                    D_u_v = H_u.sum(axis=1).reshape(1, -1)
                    D_u_e = H_u.sum(axis=0).reshape(1, -1)
                    arr1, arr2 = np.where(D_u_v==0)
                    for i in range(len(arr1)):
                        D_u_v[arr1[i], arr2[i]] = 1
                    arr1, arr2 = np.where(D_u_e==0)
                    for i in range(len(arr1)):
                        D_u_e[arr1[i], arr2[i]] = 1
                    temp1 = (H_u.transpose().multiply(np.sqrt(1.0/D_u_v))).transpose()      
                    temp2 = temp1.transpose()                                                 
                    A_u = temp1.multiply(1.0/D_u_e).dot(temp2)                                                        
                    print('A_u', A_u.get_shape())
                    D_i_v = H_i.sum(axis=1).reshape(1, -1)
                    D_i_e = H_i.sum(axis=0).reshape(1, -1)
                    arr1, arr2 = np.where(D_i_v==0)
                    for i in range(len(arr1)):
                        D_i_v[arr1[i], arr2[i]] = 1
                    arr1, arr2 = np.where(D_i_e==0)
                    for i in range(len(arr1)):
                        D_i_e[arr1[i], arr2[i]] = 1
                    temp1 = (H_i.transpose().multiply(np.sqrt(1.0 / D_i_v))).transpose()    
                    temp2 = temp1.transpose()                                               
                    A_i = temp1.multiply(1.0 / D_i_e).dot(temp2)                                                                                                     
                    print('A_i', A_i.get_shape())
                    if world.config['dropadj'] != 1:
                        A_u = drop_scipy_matrix(A_u, keep_prob=world.config['dropadj'])
                        A_i = drop_scipy_matrix(A_i, keep_prob=world.config['dropadj'])
                    temp_Graph[:self.n_users, :self.n_users] = A_u
                    temp_Graph[self.n_users:, self.n_users:] = A_i  
                    temp_Graph = temp_Graph.tocoo()
                    values = temp_Graph.data
                    indices = np.vstack((temp_Graph.row, temp_Graph.col))
                    i = torch.LongTensor(indices)
                    v = torch.FloatTensor(values)
                    shape = temp_Graph.shape
                    self.Graph = torch.sparse.FloatTensor(i, v, torch.Size(shape))
                    print("self.Graph", type(self.Graph))
                    torch.save(self.Graph, self.path + '/s_pre_adj_mat_HyperGraph2_keepprob{}.pt'.format(world.config['dropadj']))
                    print('save norm_adj in' + self.path + '/s_pre_adj_mat_HyperGraph2_keepprob{}.pt'.format(world.config['dropadj']))
                    self.Graph = self.Graph.coalesce().to(world.device)
                    return self.Graph
    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        return user
    
    def switch2test(self):
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)

class newLoader(BasicDataset):
    def __init__(self,config = world.config,path="../data/AMusic"):
        if world.dataset == 'lastfm1':
            path = "../data/lastfm"
        print("Use MYDATALOADER!!")
        cprint(f'loading [{path}]')
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if l[1] != '':
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        testUniqueUsers.append(uid)
                        testUser.extend([uid] * len(items))
                        testItem.extend(items)
                        self.m_item = max(self.m_item, max(items))
                        self.n_user = max(self.n_user, uid)
                        self.testDataSize += len(items)
        self.m_item += 1    
        self.n_user += 1   
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()    
        self.users_D[self.users_D == 0.] = 1                               
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()    
        self.items_D[self.items_D == 0.] = 1.                               
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                if world.config['load_adj'] == 'simple_adj':
                    print('====>load simple adj...')
                    pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                    print("successfully loaded s_pre_adj_mat")
                    norm_adj = pre_adj_mat
                    H_norm_adj = None
                elif world.config['load_adj'] == 'H_adj':
                    print('====>load H_adj')
                    if world.config['Hadj'] == 1:
                        # H_pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_HyperGraph.npz')
                        H_pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_HyperGraph1_keepprob{}.npz'.format(world.config['dropadj']))
                    if world.config['Hadj'] == 2:
                        H_pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_HyperGraph2_keepprob{}.npz'.format(world.config['dropadj']))
                    print("successfully loaded s_pre_adj_mat_HyperGraph")
                    norm_adj = None
                    H_norm_adj = H_pre_adj_mat
            except : 
                if world.config['load_adj'] == 'simple_adj':  
                    print("generating adjacency matrix...")
                    s = time()
                    adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                    adj_mat = adj_mat.tolil()
                    R = self.UserItemNet.tolil()
                    adj_mat[:self.n_users, self.n_users:] = R
                    adj_mat[self.n_users:, :self.n_users] = R.T
                    adj_mat = adj_mat.todok()
                    rowsum = np.array(adj_mat.sum(axis=1))
                    d_inv = np.power(rowsum + 1e-5, -0.5).flatten()
                    d_inv[np.isinf(d_inv)] = 0.
                    d_mat = sp.diags(d_inv)
                    norm_adj = d_mat.dot(adj_mat)
                    norm_adj = norm_adj.dot(d_mat)
                    norm_adj = norm_adj.tocsr()
                    end = time()
                    print(f"costing {end-s}s, saved norm_mat...")
                    sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
                if world.config['load_adj'] == 'H_adj':
                    s = time()
                    print('------------------------generate H_adj------------------------')
                    if world.config['Hadj'] == 1:
                        print("generate Hadj in one-hop")
                        R = self.UserItemNet.tolil() 
                        adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                        adj_mat = adj_mat.tolil()
                        H_u = R
                        H_i = R.transpose()
                        D_u_v = H_u.sum(axis=1).reshape(1, -1)
                        D_u_e = H_u.sum(axis=0).reshape(1, -1)
                        arr1, arr2 = np.where(D_u_v==0)
                        for i in range(len(arr1)):
                            D_u_v[arr1[i], arr2[i]] = 1
                        arr1, arr2 = np.where(D_u_e==0)
                        for i in range(len(arr1)):
                            D_u_e[arr1[i], arr2[i]] = 1
                        temp1 = (H_u.transpose().multiply(np.sqrt(1.0/D_u_v))).transpose()     
                        temp2 = temp1.transpose()                                                     
                        A_u = temp1.multiply(1.0/D_u_e).dot(temp2)                             
                        R_u = A_u.tocsr()                                                      
                        print('R_u', R_u.get_shape())
                        D_i_v = H_i.sum(axis=1).reshape(1, -1)
                        D_i_e = H_i.sum(axis=0).reshape(1, -1)
                        arr1, arr2 = np.where(D_i_v==0)
                        for i in range(len(arr1)):
                            D_i_v[arr1[i], arr2[i]] = 1
                        arr1, arr2 = np.where(D_i_e==0)
                        for i in range(len(arr1)):
                            D_i_e[arr1[i], arr2[i]] = 1
                        temp1 = (H_i.transpose().multiply(np.sqrt(1.0 / D_i_v))).transpose()    
                        temp2 = temp1.transpose()                                                      
                        A_i = temp1.multiply(1.0 / D_i_e).dot(temp2)
                        R_i = A_i.tocsr()                                                                 
                        print('R_i', R_i.get_shape())
                        adj_mat[:self.n_users, self.n_users:] = R
                        adj_mat[self.n_users:, :self.n_users] = R.T
                        if world.config['dropadj'] != 1:
                            R_u = drop_scipy_matrix(R_u, keep_prob=world.config['dropadj'])
                            R_i = drop_scipy_matrix(R_i, keep_prob=world.config['dropadj'])
                        adj_mat[:self.n_users, :self.n_users] = R_u
                        adj_mat[self.n_users:, self.n_users:] = R_i
                        H_adj_mat = adj_mat.todok()
                        rowsum = np.array(H_adj_mat.sum(axis=1))
                        d_inv = np.power(rowsum + 1e-5, -0.5).flatten()
                        d_inv[np.isinf(d_inv)] = 0.
                        d_mat = sp.diags(d_inv)
                        H_norm_adj = d_mat.dot(H_adj_mat)
                        H_norm_adj = H_norm_adj.dot(d_mat)
                        H_norm_adj = H_norm_adj.tocsr()
                        print('norm_adj', H_norm_adj.get_shape())
                        end = time()
                        print(f"costing {end-s}s, saved norm_mat...")
                        sp.save_npz(self.path + '/s_pre_adj_mat_HyperGraph1_keepprob{}.npz'.format(world.config['dropadj']), H_norm_adj)
                        print('save norm_adj in' + self.path + '/s_pre_adj_mat_HyperGraph1_keepprob{}.npz'.format(world.config['dropadj']))
                    if world.config['Hadj'] == 2:
                        print("generate Hadj in two-hop")
                        Hyper_part1 = R.transpose().dot(R)                                  # H_u = H || ( H ( HT H ) )
                        Hyper_temp1 = R.dot(Hyper_part1)                                    # ( H ( HT H ) ) = Hyper_temp1 
                        Hyper_part2 = R.dot(R.transpose())                                  # H_i = HT || ( HT ( H HT ) )
                        Hyper_temp2 = R.transpose().dot(Hyper_part2)                        # ( HT ( H HT ) ) = Hyper_temp2 
                        Hyper_temp1 = Hyper_temp1.A
                        Hyper_temp2 = Hyper_temp2.A
                        R = R.A
                        H_u = np.concatenate((R, Hyper_temp1), axis=1)                      # H_u = u,i || Hyper_temp1 -> u, 2i
                        H_i = np.concatenate((R.transpose(), Hyper_temp2), axis=1)          # H_i = i, u || Hyper_temp2 -> i, 2u                                    
                        H_u = sp.csr_matrix(H_u).tocsr()
                        H_i = sp.csr_matrix(H_i).tocsr()
                        R = sp.csr_matrix(R).tocsr() 
                        D_u_v = H_u.sum(axis=1).reshape(1, -1)
                        D_u_e = H_u.sum(axis=0).reshape(1, -1)
                        arr1, arr2 = np.where(D_u_v==0)
                        for i in range(len(arr1)):
                            D_u_v[arr1[i], arr2[i]] = 1
                        arr1, arr2 = np.where(D_u_e==0)
                        for i in range(len(arr1)):
                            D_u_e[arr1[i], arr2[i]] = 1
                        temp1 = (H_u.transpose().multiply(np.sqrt(1.0/D_u_v))).transpose()      # D_uv(-1/2) * H_u
                        temp2 = temp1.transpose()                                                     
                        A_u = temp1.multiply(1.0/D_u_e).dot(temp2)                              # D_uv(-1/2) * H_u * D_ue(-1) * H_uT * D_uv(-1/2)
                        R_u = A_u.tocsr()                                                           
                        D_i_v = H_i.sum(axis=1).reshape(1, -1)
                        D_i_e = H_i.sum(axis=0).reshape(1, -1)
                        arr1, arr2 = np.where(D_i_v==0)
                        for i in range(len(arr1)):
                            D_i_v[arr1[i], arr2[i]] = 1
                        arr1, arr2 = np.where(D_i_e==0)
                        for i in range(len(arr1)):
                            D_i_e[arr1[i], arr2[i]] = 1
                        temp1 = (H_i.transpose().multiply(np.sqrt(1.0 / D_i_v))).transpose()    
                        temp2 = temp1.transpose()                                                           
                        A_i = temp1.multiply(1.0 / D_i_e).dot(temp2)
                        R_i = A_i.tocsr()                                                                   
                        adj_mat[:self.n_users, self.n_users:] = R
                        adj_mat[self.n_users:, :self.n_users] = R.T
                        if world.config['dropadj'] != 1:
                            R_u = drop_scipy_matrix(R_u, keep_prob=world.config['dropadj'])
                            R_i = drop_scipy_matrix(R_i, keep_prob=world.config['dropadj'])
                        print("R_u", type(R_u), R_u.get_shape())
                        print("R_i", type(R_i), R_i.get_shape())
                        adj_mat[:self.n_users, :self.n_users] = R_u
                        adj_mat[self.n_users:, self.n_users:] = R_i
                        H_adj_mat = adj_mat.todok()
                        rowsum = np.array(H_adj_mat.sum(axis=1))
                        d_inv = np.power(rowsum + 1e-5, -0.5).flatten()
                        d_inv[np.isinf(d_inv)] = 0.
                        d_mat = sp.diags(d_inv)
                        H_norm_adj = d_mat.dot(H_adj_mat)
                        H_norm_adj = H_norm_adj.dot(d_mat)
                        H_norm_adj = H_norm_adj.tocsr()
                        print('norm_adj', H_norm_adj.get_shape())
                        end = time()
                        print(f"costing {end-s}s, saved norm_mat...")
                        sp.save_npz(self.path + '/s_pre_adj_mat_HyperGraph2_keepprob{}.npz'.format(world.config['dropadj']), H_norm_adj)
                        print('save norm_adj in' + self.path + '/s_pre_adj_mat_HyperGraph2_keepprob{}.npz'.format(world.config['dropadj']))
            if world.config['load_adj'] == 'simple_adj':
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the simple matrix")
                return self.Graph

            else:
                self.H_Graph = self._convert_sp_mat_to_sp_tensor(H_norm_adj)
                self.H_Graph = self.H_Graph.coalesce().to(world.device)
                print("don't split split H_matrix")
                return self.H_Graph
                

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

if __name__ == "__main__":
    dataset = newLoader(path="data/"+world.dataset)
    graph = dataset.getSparseGraph()
