# tabtransformer for regression task in rel-movielens1M
# Paper: Xin Huang, Ashish Khetan, Milan Cvitkovic, Zohar Karnin TabTransformer: Tabular Data Modeling Using Contextual Embeddings
# Arxiv: https://arxiv.org/abs/2012.06678
# MAE: 0.9599254488945007
# Runtime: 28.9525s (on a single GPU)
# Cost: $0
# Description: apply tabtransformer to rel-movielenslm, classification
# Usage: python train.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
import time
current_path = os.path.dirname(__file__)
from tab_transformer import TabTransformer
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer


## 1. Start clock
t_start = time.time()
np.random.seed(int(t_start))


## 2. Load data
train_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/ratings/train.csv')
train_rating = train_df['Rating'].values
train_df = train_df.drop(columns=['Timestamp','Rating'])
test_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/ratings/test.csv')
test_rating = test_df['Rating'].values
test_df = test_df.drop(columns=['Timestamp','Rating'])
validation_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/ratings/validation.csv')
validation_rating = validation_df['Rating'].values
validation_df = validation_df.drop(columns=['Timestamp','Rating'])
user = pd.read_csv(
    current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/users.csv')
movies = pd.read_csv(
    current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/movies.csv')

## 3. prepare categorical features
movie_genres = movies['Genre'].str.get_dummies(sep='|')
movies = movies[["MovielensID", "Title", 'Year', "Certificate"]]
movies = pd.concat([movies, movie_genres], axis=1)

user = user[['UserID','Gender','Age','Occupation']]
mapping = {'F': 0, 'M': 1}

user['Gender'] = user['Gender'].replace(mapping)
nunique = user.nunique()

def process_data(data):
    nunique = data.nunique()
    types = data.dtypes

    categorical_columns = []
    categorical_dims = {}
    for col in data.columns:
        if types[col] == 'object' or nunique[col] < 200:
            # print(col, data[col].nunique())
            data[col] = data[col].astype(str)
            l_enc = LabelEncoder()
            data[col] = data[col].fillna("VV_likely")
            data[col] = l_enc.fit_transform(data[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        elif (col in ['MovielensID']):
            categorical_columns.append(col)
            categorical_dims[col] = data[col].nunique()
        else:
            data.fillna(data[col].mean(), inplace=True)
    return data, categorical_columns, categorical_dims

movies, categorical_columns, categorical_dims = process_data(movies)

def merge_data(data):
    data_movies = pd.merge(
        data, movies, left_on='MovieID', right_on='MovielensID')

    data_movies_users = pd.merge(data_movies, user, on='UserID')

    
    data_movies_users = data_movies_users.drop(columns=['MovieID'])
    return data_movies_users

categorical_columns = ['UserID'] + categorical_columns + ['Gender', 'Age', 'Occupation']
categorical_dims['UserID'] = user['UserID'].max()+1
categorical_dims['MovielensID'] = movies['MovielensID'].max()+1
categorical_dims['Gender'] = user['Gender'].max()+1
categorical_dims['Age'] = user['Age'].max()+1
categorical_dims['Occupation'] = user['Occupation'].max()+1
cat_dims = [categorical_dims[f] for f in categorical_columns]

train = merge_data(train_df)
train = train[categorical_columns]
test = merge_data(test_df)
test = test[categorical_columns]
validation = merge_data(validation_df)
validation = validation[categorical_columns]

## 4. define models
model = TabTransformer(
    classes = cat_dims, 
    cont_names=[],
    c_out = 1
)
model.to("cuda")


## 5. Define loss function and optimizer, epoch and batchsize
criterion = nn.L1Loss()  # or nn.L1Loss() for MAE
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
batch_size = 2048

## 6. cut data to batches, mainly because my computer cannot carry more than 5000 entries of data, 
## a lot of features are ditched for the same reason, for instance, we did not use embedding

def batch(batch_size, full_size, train, train_rating):
    dataloader = []
    last = 0
    for _ in range(int(full_size / 5000) + 1):
        a = min(last + batch_size, full_size)
        dataloader.append((torch.tensor(np.array(train[last:a])).to("cuda")
                           ,torch.tensor(np.array([train_rating [last: a]]).transpose()).float().to("cuda")))
        last = a
    return dataloader
dataloader_train = batch(batch_size, train.shape[0], train, train_rating)

dataloader_val = batch(batch_size, validation.shape[0], validation, validation_rating)

dataloader_test = batch(batch_size, test.shape[0], test, test_rating)

## 7. Train model
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0
    running_loss = 0.0
    num_of_batch = 0
    for train, target in dataloader_train:  # Assuming you have a DataLoader for your training dataset
        optimizer.zero_grad() 
        outputs = model(train)

        loss = criterion(outputs, target)

        loss.backward()
    
        optimizer.step()

        epoch_loss += loss.item()
        num_of_batch += 1
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/ num_of_batch}")

## 8. valid model
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        num_of_batch_val = 0
        for input, target in dataloader_val:  # Assuming you have a DataLoader for your training dataset
            optimizer.zero_grad() 
            outputs = model(input)

            loss = criterion(outputs, target)
            val_loss += loss.item()
            num_of_batch_val += 1

        print(f"Validation Loss: {val_loss/ num_of_batch_val}")
        
## 9. test model
test_loss = 0.0
num_of_batch_test = 0
for input, target in dataloader_test:  # Assuming you have a DataLoader for your training dataset
    optimizer.zero_grad() 
    outputs = model(input)

    loss = criterion(outputs, target)
    test_loss += loss.item()
    num_of_batch_test += 1

print(f"Test Loss: {test_loss/ num_of_batch_test}")
t_end = time.time()
print('Total time: ', t_end - t_start)


