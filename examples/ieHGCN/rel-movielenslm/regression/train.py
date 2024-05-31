# ieHGCN for regression task on rel-movielenslm
# Paper: Yaming Yang, Ziyu Guan, Jianxin Li, Wei Zhao, Jiangtao Cui, Quan Wang Interpretable and Efficient Heterogeneous Graph Convolutional Network 
# Arxiv: https://arxiv.org/abs/2005.13183
# test MAE: 1.0374
# Runtime: 37.473s on GPU
# Cost: N/A
# Description: apply ieHGCN to rel-movielenslm, regression
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
	current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/ratings/train.csv')
test_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/ratings/test.csv')
validation_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/ratings/validation.csv')

user = pd.read_csv(
    current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/users.csv')
movie = pd.read_csv(
    current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/movies.csv')


# 3. data preprocess for REGRESSION

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


# pandas generate a huge table with train, test, val labels 
rating_all = pd.concat([test_df, train_df, validation_df])
rating_all

# movie id to index
mmap = _get_id_mapping(movie['MovielensID'])

# user id to user index
umap = _get_id_mapping(user['UserID'])

u = [umap[_] for _ in rating_all['UserID'].values]
m = [mmap[_] for _ in rating_all['MovieID'].values]

edge_weight = np.array(rating_all['Rating'])

edge_tensor = torch.tensor(edge_weight)

u2m = torch.sparse_coo_tensor(torch.tensor([u, m]), edge_tensor)
m2u = torch.sparse_coo_tensor(torch.tensor([m, u]), edge_tensor)
adj_dict = {"u": {"m": u2m.type(torch.FloatTensor).to("cuda")}, "m": {"u": m2u.type(torch.FloatTensor).to("cuda")}}

label = {'m': edge_tensor.type(torch.LongTensor).to("cuda")}


# # find the feature of movies, load embedding 
ft = np.load(current_path + '/../../../../rllm/datasets/embeddings.npy')

mapping = {'F': 0, 'M': 1}

# # Replace values in the 'Gender' column
user['Gender'] = user['Gender'].replace(mapping)
# # find the features of users 
user_good = user[["Gender" ,"Age" ,"Occupation"]]
user_good['Age'] = user_good["Age"] / user_good["Age"].abs().max()
user_good['Occupation'] = user_good["Occupation"] / user_good["Occupation"].abs().max()
user_ft = torch.tensor(user_good.values)

ft_dict = {"m": torch.tensor(ft).type(torch.FloatTensor).to("cuda"), "u": user_ft.type(torch.FloatTensor).to("cuda")}


trainid = train_df['MovieID'].values
validid = validation_df['MovieID'].values
testid = test_df['MovieID'].values
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

	pred = x_train_a.data.cpu().argmax(1)

	y_train_a = torch.add(label['m'][idx_train], -1)

	loss_train = F.nll_loss(x_train_a, y_train_a)

	MAE_t = nn.L1Loss()(pred.float(), y_train_a.data.cpu().float())

	loss_train.backward()
	optimizer.step()



	'''///////////////// Validating ///////////////////'''

	model.eval()
	logits, _ = model(ft_dict, adj_dict)

	m_logits = F.log_softmax(logits['m'], dim=1)
	# idx_val_a = label['m']
	x_val_a = m_logits[idx_val]
	pred_val = x_val_a.data.cpu().argmax(1)
	y_val_a = torch.add(label['m'][idx_val], -1)
	MAE_v = nn.L1Loss()(pred_val.float(), y_val_a.data.cpu().float())

	
	if epoch % 1 == 0:
		print(
			  'epoch: {:3d}'.format(epoch),
			  'train loss: {:.4f}'.format(loss_train.item()),
			  'train MAE: {:.4f}'.format(MAE_t.item()),
			  'val MAE: {:.4f}'.format(MAE_v.item()),
			 )



def test():
	model.eval()
	logits, embd = model(ft_dict, adj_dict)

	m_logits = F.log_softmax(logits['m'], dim=1)
	x_test_a = m_logits[idx_test]
	pred = x_test_a.data.cpu().argmax(1)
	y_test_a = torch.add(label['m'][idx_test], -1)
	MAE_test = nn.L1Loss()(pred.float(), y_test_a.data.cpu().float())


	
	print(
		  '\n'+
  		  'test MAE: {:.4f}'.format(MAE_test.item()),
		#   'test macro f1 a: {:.4f}'.format(f1_macro_test_a.item()),
		 )

	return (MAE_test)



if __name__ == '__main__':

	cuda = True # Enables CUDA training.
	# lr = 0.03 # Initial learning rate
	lr = 0.01
	weight_decay = 5e-3 # Weight decay (L2 loss on parameters).
	type_att_size = 64 # type attention parameter dimension
	type_fusion = 'att' # mean
	
	run_num = 1
	train_percent = 0.2
	for run in range(run_num):
		# t_start = time.time()
		seed = run

		np.random.seed(seed)
		torch.manual_seed(seed)
		if cuda and torch.cuda.is_available():
			torch.cuda.manual_seed(seed)

		print('\nHGCN run: ', run)
		print('train percent: ', train_percent)
		print('seed: ', seed)
		print('type fusion: ', type_fusion)
		print('type att size: ', type_att_size)

		hid_layer_dim = [128, 32, 16, 8]  # dblp4area4057
		epochs = 150
		output_layer_shape = dict.fromkeys(ft_dict.keys(), 5)
		# print(output_layer_shape)


		layer_shape = []
		input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
		layer_shape.append(input_layer_shape)
		hidden_layer_shape = [dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in hid_layer_dim]
		layer_shape.extend(hidden_layer_shape)
		layer_shape.append(output_layer_shape)

		# 4. define Model and optimizer
		net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
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
		MAE_test = test()

		t_end = time.time()
		print('Total time: ', t_end - t_start)