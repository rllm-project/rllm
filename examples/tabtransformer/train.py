# tabtransformer for regression task in adult
# Paper: Xin Huang, Ashish Khetan, Milan Cvitkovic, Zohar Karnin TabTransformer: Tabular Data Modeling Using Contextual Embeddings
# Arxiv: https://arxiv.org/abs/2012.06678
# MAE: 0.2459743469953537
# Runtime: 52.63491082191467s (on single GPU)
# Cost: $0
# Description: apply tabtransformer to adult, regression
# Usage: python train.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
from tab_transformer import TabTransformer
from sklearn.preprocessing import LabelEncoder
import time

t_start = time.time()
np.random.seed(int(t_start))

## load data
train = pd.read_csv('F:\\rllm\\examples\\tabtransformer\\dataset\\adult.data', header=None)
target = 14

## 2. cutting dataset into train, validation, test set
if "Set" not in train.columns:
    train["Set"] = np.random.choice(
        ["train", "valid", "test"], p=[.8, .1, .1], size=(train.shape[0],))

train_indices = train[train.Set == "train"].index
valid_indices = train[train.Set == "valid"].index
test_indices = train[train.Set == "test"].index


nunique = train.nunique()
types = train.dtypes

## 3. preprocess data
categorical_columns = []
categorical_dims = {}
for col in train.columns:
    if types[col] == 'object' or nunique[col] < 200:
        # print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)


unused_feat = ['Set']

features = [col for col in range(0, 14)]
cat_idxs = [i for i, f in enumerate(features) if f in categorical_columns]
cat_dims = [categorical_dims[f]
            for i, f in enumerate(features) if f in categorical_columns]

## 4. define model
model = TabTransformer(
    classes = cat_dims, 
    cont_names = [], 
    c_out = 1
)

## 5. prepare X and y
X_train = train[cat_idxs].values[train_indices]
y_train = train[target].values[train_indices]

length = len(X_train)

X_valid = train[cat_idxs].values[valid_indices]
y_valid = train[target].values[valid_indices]

X_test = train[cat_idxs].values[test_indices]
y_test = train[target].values[test_indices]

y_train = np.transpose([y_train])
y_valid = np.transpose([y_valid])

target = torch.tensor(y_train).float()
val_target = torch.tensor(y_valid).float()
test_target = torch.tensor(y_test).float()

# 6. Define loss function and optimizer, epoch
criterion = nn.L1Loss()  # or nn.L1Loss() for MAE
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# 7. train 
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    
    running_loss = 0.0
    optimizer.zero_grad()
    x_categ = torch.tensor(np.array(X_train))     # category values, from 0 - max number of categories, in the order as passed into the constructor above
    outputs = model(x_categ)
    
    loss = criterion(outputs, target)
    loss.backward()
    
    optimizer.step()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# 8. evaluation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        x_categ = torch.tensor(np.array(X_valid))  
        outputs = model(x_categ)

        loss = criterion(outputs, val_target)
        print(f"Validation Loss: {loss.item()}")

# 9. test    
x_categ = torch.tensor(np.array(X_test))     # category values, from 0 - max number of categories, in the order as passed into the constructor above
outputs = model(x_categ)
pred = pred = np.where(outputs.cpu() > 0.5 , 1, 0)
loss = criterion(outputs, test_target )
print(f"Test Loss: {loss.item()}")
t_end = time.time()
print('Total time: ', t_end - t_start)