import sys
import numpy as np
from load_data import load_data
import torch
def sample_mask(idx, lst):
    """Create mask."""
    mask = np.zeros(lst)
    mask[idx] = 1
    return np.array(mask, dtype=bool)

def separate_data():
    data, adj, features, labels, idx_train, idx_val, idx_test = load_data('movielens-classification')
    label_origin = labels.detach().cpu().numpy()

    # move to fit
    device = torch.device("cuda")
    train_mask = sample_mask(idx_train, label_origin.shape[0])
    val_mask = sample_mask(idx_val, label_origin.shape[0])
    test_mask = sample_mask(idx_test, label_origin.shape[0])

    y_train = np.zeros(label_origin.shape)
    y_val = np.zeros(label_origin.shape)
    y_test = np.zeros(label_origin.shape)
    y_train[train_mask, :] = label_origin[train_mask, :]
    y_val[val_mask, :] = label_origin[val_mask, :]
    y_test[test_mask, :] = label_origin[test_mask, :]
    train_index = np.where(train_mask)[0]
    y_train = y_train[train_index]
    val_index = np.where(val_mask)[0]
    y_val = y_val[val_index]
    test_index = np.where(test_mask)[0]
    y_test = y_test[test_index]

    train_adj = adj[train_index, :][:, train_index]

    train_feats = features[train_index]

    test_adj = adj[idx_test, :][:, idx_test]
    test_feats = features[test_index]
    test_labels = y_test
   
    val_adj = adj[idx_val, :]
    val_feats = features
    val_labels = y_val
    
    return data, adj, features, labels, idx_train, idx_test, y_train, y_test, train_adj, test_adj, train_feats, test_feats, test_labels, val_adj, val_feats, val_labels

def get_batches(train_ind, train_labels, train_feats, train_adjs, batch_size=64, shuffle=True):
    """
    Inputs:
        train_ind: np.array
    """
    nums = train_ind.shape[0]
    if shuffle:
        np.random.shuffle(train_ind)
    i = 0
    while i < nums:
        cur_ind = train_ind[i:i + batch_size]
        #print('cur_ind',cur_ind)

        cur_labels = train_labels[cur_ind]
        cur_labels = torch.tensor(cur_labels)
        #print('cur_labels',cur_labels)
        #print(type(cur_labels))
        sampled_feats = train_feats[cur_ind]
        sampled_adjs = train_adjs[cur_ind].long()
        '''edges = torch.nonzero(sampled_adjs == 1).squeeze()
        edges = torch.transpose(edges, 0, 1).long()'''
        #sampled_adjs = train_adjs[cur_ind]
        '''# 初始化空列表来存储图中所有边的矩阵
        edges = []

        # 遍历源数据的每一个节点
        for i in range(train_adjs.size(0)):
            for j in range(train_adjs.size(1)):  # 仅遍历上三角矩阵
                # 如果关系向量中有边（值为1）,则将这两个节点的标号存储起来
                if sampled_adjs[i][j] == 1:
                    edges.append([i, j])
        edges = torch.tensor(edges)
        edges = torch.transpose(edges, 0, 1)'''

        yield cur_ind, cur_labels, sampled_feats, sampled_adjs
        i += batch_size
