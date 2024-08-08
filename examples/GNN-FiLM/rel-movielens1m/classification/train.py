# GNN-FiLM for classification task in rel-movielens1m dataset
# Paper: Brockschmidt, M. (2020, November). Gnn-film: Graph neural networks with feature-wise linear modulation. In International Conference on Machine Learning (pp. 1144-1152). PMLR.
# Test f1_score micro: 0.18624943583571535; macro: 0.1134700781400389
# Runtime: 25.3940s on a single GPU
# Cost: N/A

import time
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.nn import BatchNorm1d
from film_conv import FiLMConv
import numpy as np
from utils import separate_data, get_batches

data, adj, features, labels, idx_train, idx_test, y_train, y_test, train_adj, test_adj, train_feats, test_feats, test_labels, val_adj, val_feats, val_labels = separate_data()
input_dim = features.shape[1]
train_nums = train_adj.shape[0]
if torch.cuda.is_available():
    device = torch.device('cuda:1')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.0):
        super().__init__()
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(FiLMConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(FiLMConv(hidden_channels, hidden_channels))
        self.convs.append(FiLMConv(hidden_channels, out_channels, act=None))
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.norms.append(BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

torch.cuda.init()
model = Net(in_channels=features.shape[1], hidden_channels=320,
            out_channels=y_train.shape[1], num_layers=8,
            dropout=0.1).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(train_ind, batch_size):
    model.train()
    for batch_labels, sampled_feats, sampled_adjs in get_batches(train_ind,
                                                    y_train,train_feats,train_adj, batch_size, False):
        sampled_feats = sampled_feats.to(device)
        sampled_adjs = sampled_adjs.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        out = model(sampled_feats, sampled_adjs)
        loss = criterion(out, batch_labels)
        loss.backward()
        optimizer.step()
    return loss.item()


@torch.no_grad()
def test(test_y, test_f, test_a):
    model.eval()
    ys, preds = [], []
    if test_a.shape[0] %2 == 0:
        test_nums = test_a.shape[0]
    else:
        test_nums = test_a.shape[0]-1
    for batch_labels, sampled_feats, sampled_adjs in get_batches(np.arange(test_nums),
                                                    test_y, test_f, test_a, 2, False):
    #for data in loader:
        ys.append(batch_labels)
        out = model(sampled_feats.to(device), sampled_adjs.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    f1_micro = f1_score(y, pred, average='micro')
    f1_macro = f1_score(y, pred, average='macro')
    return f1_micro, f1_macro

t_total = time.time()
for epoch in range(1, 11):
    loss = train(np.arange(100), 2) 
    f1_micro, f1_macro = test(val_labels, val_feats, val_adj)
    value = (f1_micro+f1_macro)/2.0
    if epoch < 6 and value > 0.18:
            break
    if epoch > 5 and value > 0.175:
            break

f1_micro, f1_macro = test(y_test, test_feats, test_adj)
print(f"micro: {f1_micro}; macro: {f1_macro}")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

