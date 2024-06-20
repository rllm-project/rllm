#GCNII
#paper:Simple and Deep Graph Convolutional Networks
#Arxiv:https://arxiv.org/abs/2007.02133
#loss:0.2783
#Runtime：198.0222s(single 6G GPU)
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
    output = model(features,adj)
    #print("output",output)
    pred = np.where(output.cpu() > -1.0, 1, 0)
    loss_train = loss_func(output[idx_train], labels[idx_train])
    f1_micro_train = f1_score(labels_train, pred[idx_train.cpu()], average="micro")
    f1_macro_train = f1_score(labels_train, pred[idx_train.cpu()], average="macro")
    loss_train.backward()
    optimizer.step()

    loss_val = loss_func(output[idx_val], labels[idx_val])
    f1_micro_val = f1_score(labels_val, pred[idx_val.cpu()], average="micro")
    f1_macro_val = f1_score(labels_val, pred[idx_val.cpu()], average="macro")
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'f1_train: {:.4f} {:.4f}'.format(f1_micro_train, f1_macro_train),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'f1_val: {:.4f} {:.4f}'.format(f1_micro_val, f1_macro_val),
          'time: {:.4f}s'.format(time.time() - t))


def validate():
    model.eval()
    with torch.no_grad():
        output = model(features,adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

#def test():
    #model.load_state_dict(torch.load(checkpt_file))
    #model.eval()
    #with torch.no_grad():
        #output = model(features, adj)
        #loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        #acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        #return loss_test.item(),acc_test.item()

def test():
    model.eval()
    output = model(features, adj)
    pred = np.where(output.cpu() > -1.0, 1, 0)
    loss_test = loss_func(output[idx_test], labels[idx_test])
    f1_micro_test = f1_score(labels_test, pred[idx_test.cpu()], average="micro")
    f1_macro_test = f1_score(labels_test, pred[idx_test.cpu()], average="macro")
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "f1_test= {:.4f} {:.4f}".format(f1_micro_test, f1_macro_test))

# Train model
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()  





