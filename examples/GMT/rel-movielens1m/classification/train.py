# GMT for classification task in rel-movielens1m dataset
# Paper: Baek, J., Kang, M., & Hwang, S. J. (2021). Accurate learning of graph representations with graph multiset pooling. arXiv preprint arXiv:2102.11533.
# Test f1_score micro: 0.33439635535307516; macro: 0.0799316725838052
# Runtime: 7.1744s on a single GPU
# Cost: N/A

import os.path as osp
import time
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch.nn import Linear
import numpy as np
from utils import separate_data,get_batches
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphMultisetTransformer
import warnings
warnings.filterwarnings("ignore")
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PROTEINS')
dataset = TUDataset(path, name='PROTEINS').shuffle()

n = (len(dataset) + 9) // 10
train_dataset = dataset[2 * n:]
val_dataset = dataset[n:2 * n]
test_dataset = dataset[:n]

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

data, adj, features, labels, idx_train, idx_test, y_train, y_test, train_adj, test_adj, train_feats, test_feats, test_labels, val_adj, val_feats, val_labels = separate_data()

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(features.shape[1], 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)

        self.pool = GraphMultisetTransformer(96, k=10, heads=4)

        self.lin1 = Linear(32, 16)
        self.lin2 = Linear(16, y_train.shape[1])

    def forward(self, x0, edge_index, batch):
        x1 = self.conv1(x0, edge_index).relu()
        x2 = self.conv2(x1, edge_index).relu()
        x3 = self.conv3(x2, edge_index).relu()
        x = torch.cat([x1, x2, x3], dim=-1)

        #x = self.pool(x, batch)

        x = self.lin1(x3).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


def train(train_ind, batch_size):
    model.train()

    total_loss = 0
    for cur_ind, batch_labels, sampled_feats, sampled_adjs in get_batches(train_ind,
                                                    y_train,train_feats,train_adj, batch_size, False):
        #data = data.to(device)
        optimizer.zero_grad()
        '''print(sampled_feats.size())
        print(sampled_adjs.size())
        print(batch_labels.size())'''
        sampled_feats = sampled_feats.to(device)
        sampled_adjs = sampled_adjs.to(device)
        batch_labels = batch_labels.to(device)
        out = model(sampled_feats, sampled_adjs, cur_ind)

        loss = F.cross_entropy(out, batch_labels)
        loss.backward()
        total_loss += float(loss)
        optimizer.step()
    return total_loss / 100

@torch.no_grad()
def test():
    model.eval()
    ys, preds = [], []
    test_nums = test_adj.shape[0]-1
    for cur_ind, batch_labels, sampled_feats, sampled_adjs in get_batches(np.arange(test_nums),
                                                    y_test, test_feats, test_adj, 2, False):
        sampled_feats = sampled_feats.to(device)
        sampled_adjs = sampled_adjs.to(device)
        out = model(sampled_feats, sampled_adjs, cur_ind)
        ys.append(batch_labels)
        preds.append((out > 0).cpu())
    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()

    f1_micro = f1_score(y, pred, average='micro')
    f1_macro = f1_score(y, pred, average='macro')
    return f1_micro, f1_macro

t_total = time.time()
for epoch in range(1, 11):
    start = time.time()
    train_loss = train(np.arange(100), 2)
f1_micro, f1_macro = test()
print(f"micro: {f1_micro}; macro: {f1_macro}")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
