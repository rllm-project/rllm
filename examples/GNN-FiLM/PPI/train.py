# GNN-FiLM for classification task in PPI dataset
# Paper: Brockschmidt, M. (2020, November). Gnn-film: Graph neural networks with feature-wise linear modulation. In International Conference on Machine Learning (pp. 1144-1152). PMLR.
# Test f1_score micro: 0.6817220020774823; macro: 0.5575759971985581
# Runtime: 11.3626s on a single GPU
# Cost: N/A

import os.path as osp
import time
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.nn import BatchNorm1d

from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
#from torch_geometric.nn import FiLMConv
from film_conv import FiLMConv
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')

test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


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


if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = Net(in_channels=train_dataset.num_features, hidden_channels=320,
            out_channels=train_dataset.num_classes, num_layers=4,
            dropout=0.1).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        '''print('x',data.x.size())
        print('y',data.y.size())
        print('edge_index',data.edge_index[0])
        print('edge_index',data.edge_index.size())'''
        data = data.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    f1_micro = f1_score(y, pred, average='micro')
    f1_macro = f1_score(y, pred, average='macro')
    return f1_micro, f1_macro

t_total = time.time()
for epoch in range(1, 16):
    loss = train() 
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f} ')

f1_micro, f1_macro = test(test_loader)
print(f"micro: {f1_micro}; macro: {f1_macro}")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
