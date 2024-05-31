# ieHGCN for classification task in rel-movielenslm
# Paper: Yaming Yang, Ziyu Guan, Jianxin Li, Wei Zhao, Jiangtao Cui, Quan Wang Interpretable and Efficient Heterogeneous Graph Convolutional Network 
# Arxiv: https://arxiv.org/abs/2005.13183
# test micro f1 a: 0.3363 test macro f1 a: 0.0341 (depend on random seed, but if you run it a couple of times you should be able to recreate the result)
# Runtime: 62.993s on GPU (with data loading) 56.977s on GPU (without data loading also depend on seed, sometimes the plateau is reached much faster, around 20s, 300 epochs are chosen to ensure the likelihood of recreating the result)
# Cost: N/A
# Description: apply ieHGCN to rel-movielenslm, classification
# Usage: python train.py

# 1. import
import pandas as pd
import numpy as np
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
import os
current_path = os.path.dirname(__file__)


warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

import time
from model import HGCN

t_start = time.time()

# 2. load data
train_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/classification/movies/train.csv')
test_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/classification/movies/test.csv')
validation_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/classification/movies/validation.csv')

user = pd.read_csv(
    current_path + '/../../../../rllm/datasets/rel-movielens1m/classification/users.csv')
rating = pd.read_csv(
    current_path + '/../../../../rllm/datasets/rel-movielens1m/classification/ratings.csv')


# 3. data preprocess for CLASSIFICATION

def _get_id_mapping(ids):
    r"""
    Create an index mapping from index List `ids`.
    """
    mapping, cnt = {}, 0
    for _ in ids:
        assert (_ not in mapping)
        mapping[_] = cnt
        cnt += 1
    return mapping


movie_all = pd.concat([test_df, train_df, validation_df])

mmap = _get_id_mapping(movie_all['MovielensID'])
umap = _get_id_mapping(user['UserID'])

u = [umap[_] for _ in rating['UserID'].values]
m = [mmap[_] for _ in rating['MovieID'].values]

edge_weight = rating['Rating']

u2m = torch.sparse_coo_tensor(torch.tensor([u, m]), torch.tensor(edge_weight))
m2u = torch.sparse_coo_tensor(torch.tensor([m, u]), torch.tensor(edge_weight))
adj_dict = {"u": {"m": u2m.type(torch.FloatTensor).to("cuda")}, "m": {"u": m2u.type(torch.FloatTensor).to("cuda")}}

# find the adj_dict user->movie same to label
labels = torch.tensor(np.array(movie_all['Genre'].str.get_dummies('|'))) #index is the movie id
label = {'m': labels.type(torch.FloatTensor).to("cuda")}


ft = np.load(current_path + '/../../../../rllm/datasets/embeddings.npy')

mapping = {'F': 0, 'M': 1}

# Replace values in the 'Gender' column
user['Gender'] = user['Gender'].replace(mapping)
# find the features of users 
user_good = user[["Gender" ,"Age" ,"Occupation"]]
# print(user_good.values)
user_good['Age'] = user_good["Age"] / user_good["Age"].abs().max()
user_good['Occupation'] = user_good["Occupation"] / user_good["Occupation"].abs().max()
user_ft = torch.tensor(user_good.values)

ft_dict = {"m": torch.tensor(ft).type(torch.FloatTensor).to("cuda"), "u": user_ft.type(torch.FloatTensor).to("cuda")}


trainid = train_df['MovielensID'].values
validid = validation_df['MovielensID'].values
testid = test_df['MovielensID'].values
idx_train = torch.LongTensor([mmap[i] for i in trainid])
idx_val = torch.LongTensor([mmap[i] for i in validid])
idx_test = torch.LongTensor([mmap[i] for i in testid])


# 3. define train and test function

def train(epoch):

	model.train()
	optimizer.zero_grad()
	logits, _ = model(ft_dict, adj_dict)


	m_logits = F.log_softmax(logits['m'], dim=1)
	x_train_a = m_logits[idx_train]
	pred = np.where(x_train_a.cpu() > -0.5 , 1, 0)
	y_train_a = label['m'][idx_train]

	loss_train = F.cross_entropy(x_train_a, y_train_a)
	f1_micro_train_a = f1_score(y_train_a.data.cpu(), pred, average='micro')
	f1_macro_train_a = f1_score(y_train_a.data.cpu(), pred, average='macro')

	loss_train.backward()
	optimizer.step()



	'''///////////////// Validating ///////////////////'''

	model.eval()
	logits, _ = model(ft_dict, adj_dict)

	m_logits = F.log_softmax(logits['m'], dim=1)
	x_val_a = m_logits[idx_val]
	pred_val = np.where(x_val_a.cpu() > -0.5 , 1, 0)
	y_val_a = label['m'][idx_val]
	f1_micro_val_a = f1_score(y_val_a.data.cpu(), pred_val, average='micro')
	f1_macro_val_a = f1_score(y_val_a.data.cpu(), pred_val, average='macro')

	
	if epoch % 1 == 0:
		print(
			  'epoch: {:3d}'.format(epoch),
			  'train loss: {:.4f}'.format(loss_train.item()),
			  'train micro f1 a: {:.4f}'.format(f1_micro_train_a.item()),
			  'train macro f1 a: {:.4f}'.format(f1_macro_train_a.item()),
			  'val micro f1 a: {:.4f}'.format(f1_micro_val_a.item()),
			  'val macro f1 a: {:.4f}'.format(f1_macro_val_a.item()),
			 )



def test():
	model.eval()
	logits, embd = model(ft_dict, adj_dict)

	m_logits = F.log_softmax(logits['m'], dim=1)
	x_test_a = m_logits[idx_test]
	pred = np.where(x_test_a.cpu() > -0.5 , 1, 0)
	y_test_a = label['m'][idx_test]
	f1_micro_test_a = f1_score(y_test_a.data.cpu(), pred, average='micro')
	f1_macro_test_a = f1_score(y_test_a.data.cpu(), pred, average='macro')

	
	print(
		  '\n'+
  		  'test micro f1 a: {:.4f}'.format(f1_micro_test_a.item()),
		  'test macro f1 a: {:.4f}'.format(f1_macro_test_a.item()),
		 )

	return (f1_micro_test_a, f1_macro_test_a)



if __name__ == '__main__':

	cuda = True # Enables CUDA training.
	# lr = 0.03 # Initial learning rate
	lr = 0.01
	weight_decay = 5e-4 # Weight decay (L2 loss on parameters).
	type_att_size = 64 # type attention parameter dimension
	type_fusion = 'att' # mean
	
	run_num = 1
	train_percent = 0.2
	for run in range(run_num):
		# t_start = time.time()
		seed = 1
		np.random.seed(seed)
		torch.manual_seed(seed)
		if cuda and torch.cuda.is_available():
			torch.cuda.manual_seed(seed)

		print('\nHGCN run: ', run)
		print('train percent: ', train_percent)
		print('seed: ', seed)
		print('type fusion: ', type_fusion)
		print('type att size: ', type_att_size)

		hid_layer_dim = [128,64,32,18]  # dblp4area4057
		epochs = 250
		output_layer_shape = dict.fromkeys(ft_dict.keys(), 18)



		layer_shape = []
		input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
		layer_shape.append(input_layer_shape)
		hidden_layer_shape = [dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in hid_layer_dim]

		layer_shape.extend(hidden_layer_shape)
		layer_shape.append(output_layer_shape)

		net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])

		# 4. define Model and optimizer
		model = HGCN(
					net_schem=net_schema,
					layer_shape=layer_shape,
					label_keys=list(label.keys()),
					type_fusion=type_fusion,
					type_attention_size=type_att_size,
					)
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

		# 5. load data to cuda
		if cuda and torch.cuda.is_available():
			model.cuda()

			for k in ft_dict:
				ft_dict[k] = ft_dict[k].cuda()
			for k in adj_dict:
				for kk in adj_dict[k]:
					adj_dict[k][kk] = adj_dict[k][kk].cuda()
			for k in label:
				for i in range(len(label[k])):
					label[k][i] = label[k][i].cuda()

		# 6. train
		for epoch in range(epochs):
			train(epoch)

		# 7. test
		(micro_f1, macro_f1) = test()

		t_end = time.time()
		print('Total time: ', t_end - t_start)