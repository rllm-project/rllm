#GCNII
#paper:Simple and Deep Graph Convolutional Networks
#Arxiv:https://arxiv.org/abs/2007.02133
#loss:0.2783
#Runtime：161.5396s(single 6G GPU)
#Usage:python train.py

from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import *
import sys 
sys.path.append("../../../../rllm/dataloader")
from load_data import load_data
# sys.path.append("D:/rllm/examples/gcn/rel-movielens1m/classification")
# from models import *

import uuid
from sklearn.metrics import f1_score

t_total = time.time()
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=30, help='Random seed.')
parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6 , help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.05, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
#args = parser.parse_args()
#random.seed(args.seed)
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
#torch.cuda.manual_seed(args.seed)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cuda' if args.cuda else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)

#adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.data)
data, adj, features, labels, idx_train, idx_val, idx_test = load_data('movielens-classification', device=device)
from random import random
adj_dense = adj.to_dense()
adj_drop = torch.zeros_like(adj_dense).to(device)
adj_drop = adj_drop.to_sparse().coalesce()

# print(data.x["movie"])
#labels = labels.argmax(dim=-1)
labels_train = labels.cpu()[idx_train.cpu()]
labels_val = labels.cpu()[idx_val.cpu()]
labels_test = labels.cpu()[idx_test.cpu()]
# torch.savetxt("features.txt",features)
#注释了59 61
#features = features.to(device)
# print(features)
#adj = adj.to(device)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
# print(cudaid,checkpt_file)
# print(labels.shape[1])
model = GCNII(nfeat=features.shape[1],
                nlayers=args.layer,
                nhidden=args.hidden,
                # nclass=int(labels.max()) + 1,
                nclass=labels.shape[1],
                dropout=args.dropout,
                lamda = args.lamda, 
                alpha=args.alpha,
                variant=args.variant,
                train = True).to(device)

optimizer = optim.Adam([
                        {'params':model.params1,'weight_decay':args.wd1},
                        {'params':model.params2,'weight_decay':args.wd2},
                        ],lr=args.lr)


loss_func = nn.BCEWithLogitsLoss()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # print(output[[0, 1, 2, 3, 4, 5]])
    loss_train = loss_func(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_val = loss_func(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = loss_func(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
