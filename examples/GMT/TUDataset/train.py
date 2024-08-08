# GMT for classification task in TUDataset dataset
# Paper: Baek, J., Kang, M., & Hwang, S. J. (2021). Accurate learning of graph representations with graph multiset pooling. arXiv preprint arXiv:2102.11533.
# Test f1_score micro: 0.43037974683544306; macro: 0.15178571428571427
# Runtime: 6.8501s on a single GPU
# Cost: N/A

import os.path as osp
import time
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch.nn import Linear

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


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(dataset.num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)

        self.pool = GraphMultisetTransformer(96, k=10, heads=4)

        self.lin1 = Linear(96, 16)
        self.lin2 = Linear(16, dataset.num_classes)

    def forward(self, x0, edge_index, batch):
        x1 = self.conv1(x0, edge_index).relu()
        x2 = self.conv2(x1, edge_index).relu()
        x3 = self.conv3(x2, edge_index).relu()
        x = torch.cat([x1, x2, x3], dim=-1)

        x = self.pool(x, batch)

        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        total_loss += data.num_graphs * float(loss)
        optimizer.step()
    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    ys, preds = [], []
    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        ys.append(data.y.cpu().numpy())
        max_values, _ = torch.max(out, dim=1)
        preds.append(out.argmax(dim=-1).cpu().numpy())
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
    f1_micro = f1_score(ys, preds, average='micro')
    f1_macro = f1_score(ys, preds, average='macro')
    return f1_micro, f1_macro
    #return total_correct / len(loader.dataset)


t_total = time.time()
for epoch in range(1, 31):
    start = time.time()
    train_loss = train()
    val_acc = test(val_loader)
    
f1_micro, f1_macro = test(test_loader)
    
print(f"micro: {f1_micro}; macro: {f1_macro}")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
