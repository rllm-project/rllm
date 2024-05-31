# tabtransformer for classification task in rel-movielens1M
# Paper: Xin Huang, Ashish Khetan, Milan Cvitkovic, Zohar Karnin TabTransformer: Tabular Data Modeling Using Contextual Embeddings
# Arxiv: https://arxiv.org/abs/2012.06678
# test micro f1 a: 0.2098 test macro f1 a: 0.0657
# Runtime: 27.065s (on a single GPU)
# Cost: $0
# Description: apply tabtransformer to rel-movielenslm, regression
# Usage: python train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.optim as optim
import time
from sklearn.metrics import f1_score
from tab_transformer import TabTransformer
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer


## 1. Start clock

t_start = time.time()
np.random.seed(int(t_start))

## 2. load data
current_path = 'f:\\ie\\example\\tabtransformer\\rel-movielenslm\\classification'
train_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/classification/movies/train.csv')
train_df["Set"] = 0
test_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/classification/movies/test.csv')
test_df["Set"] = 2
validation_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/classification/movies/validation.csv')
validation_df["Set"] = 1


## 3. get target
train_genres = train_df['Genre'].str.get_dummies(sep='|')
validation_genres = validation_df['Genre'].str.get_dummies(sep='|')
test_genres = test_df['Genre'].str.get_dummies(sep='|')


## 4. extract features
movie_all = pd.concat([test_df, train_df, validation_df]).drop(columns=["Year", "Languages", "Url", "Cast", "Runtime", "MovielensID"])
model = SentenceTransformer('F:\\rllm\\examples\\tabtransformer\\rel-movielenslm\\classification\\all-MiniLM-L6-v2')
movie_rate = movie_all['Certificate'].str.get_dummies()
# movie_director = movie_all['Director'].str.get_dummies(sep=",")
sentences = (movie_all["Title"]+movie_all["Plot"]).fillna(value="").tolist()
embeddings = model.encode(sentences)
movie = pd.concat([movie_all, movie_rate], axis=1).join(pd.DataFrame(embeddings)).drop(columns=["Title", "Genre", "Director", "Certificate", "Plot"])

train = movie[movie["Set"]==0]
test = movie[movie["Set"]==2]
validation = movie[movie["Set"]==1]

categorical_columns = movie_rate.columns.tolist()
cont_names = [col for col in range(0, 384)]
cat_dims = [2 for _ in categorical_columns]

train_cat = train[categorical_columns]
test_cat = test[categorical_columns]
validation_cat = validation[categorical_columns]
train_cont = train[cont_names]
test_cont = test[cont_names]
validation_cont = validation[cont_names]


## 5. Define model (cat_dim could be larger, for instance, movie director is not used for my computer cannot carry that many features)
model = TabTransformer(
    classes = cat_dims, 
    cont_names = cont_names, 
    c_out = 18
)


## 6. learning rate, weight decay, optimizer, epoch
lr = 0.01 # Initial learning rate.c
weight_decay = 5e-4 # Weight decay (L2 loss on parameters).
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
num_epochs = 150


## 7. train
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()
    x_categ = torch.tensor(np.array(train_cat)).int()    # category values, from 0 - max number of categories, in the order as passed into the constructor above
    x_cont = torch.tensor(np.array(train_cont)).float()
    y = torch.tensor(np.array(train_genres)).float()
    outputs = model(x_categ, x_cont)

    logits = F.log_softmax(outputs, dim=1)
    pred = np.where(logits.cpu() > -1.0 , 1, 0)
    loss_train = F.cross_entropy(logits, y)

    f1_micro_train = f1_score(y.data.cpu(), pred, average='micro')
    f1_macro_train = f1_score(y.data.cpu(), pred, average='macro')

    loss_train.backward()
    optimizer.step()


## 7. evaluation
    model.eval()
    with torch.no_grad():
        x_categ_val = torch.tensor(np.array(validation_cat)).int()    # category values, from 0 - max number of categories, in the order as passed into the constructor above
        x_cont_val = torch.tensor(np.array(validation_cont)).float()
        y_val = torch.tensor(np.array(validation_genres)).float()
        outputs = model(x_categ_val, x_cont_val)
    

        logits_val = F.log_softmax(outputs, dim=1)
        pred_val = np.where(logits_val.cpu() > -1.0 , 1, 0)

        f1_micro_val = f1_score(y_val.data.cpu(), pred_val, average='micro')
        f1_macro_val = f1_score(y_val.data.cpu(), pred_val, average='macro')

        if epoch % 1 == 0:
            print(
			  'epoch: {:3d}'.format(epoch),
			  'train loss: {:.4f}'.format(loss_train.item()),
			  'train micro f1 a: {:.4f}'.format(f1_micro_train.item()),
			  'train macro f1 a: {:.4f}'.format(f1_macro_train.item()),
			  'val micro f1 a: {:.4f}'.format(f1_micro_val.item()),
			  'val macro f1 a: {:.4f}'.format(f1_macro_val.item()),
			)

## 8. test
model.eval()
x_categ_test = torch.tensor(np.array(test_cat)).int()    # category values, from 0 - max number of categories, in the order as passed into the constructor above
x_cont_test = torch.tensor(np.array(test_cont)).float()
y_test = torch.tensor(np.array(test_genres)).float()
outputs = model(x_categ_test, x_cont_test)

logits_test = F.log_softmax(outputs, dim=1)
pred_test = np.where(logits_test.cpu() > -1.0 , 1, 0)

f1_micro_test = f1_score(y_test.data.cpu(), pred_test, average='micro')
f1_macro_test = f1_score(y_test.data.cpu(), pred_test, average='macro')

print(
        '\n'+
        'test micro f1 a: {:.4f}'.format(f1_micro_test.item()),
        'test macro f1 a: {:.4f}'.format(f1_macro_test.item()),
        )

t_end = time.time()
print('Total time: ', t_end - t_start)