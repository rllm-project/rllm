# Linear Regression pipeline for regression task in rel-movielens1M
# Paper:
# MAE: 0.7395652008568703
# Runtime: 7.861s
# Cost: 0
# Description: Use mean user rating and mean movie rating to predict THE USER's rating of THE movie
# Usage: python


import pandas as pd
import numpy as np
import math
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

## 1. Start clock
t_start = time.time()

## 2. Load Data
current_path = 'F:\\rllm2\\examples\\group_9_method\\rel-movielenslm\\regression'
train_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/ratings/train.csv')
train_rating = train_df['Rating']
test_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/ratings/test.csv')
test_rating = test_df['Rating']
validation_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/ratings/validation.csv')
validation_rating = validation_df['Rating']
train_df = pd.concat([train_df, validation_df])
train_rating = pd.concat([train_rating, validation_rating])
user = pd.read_csv(
    current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/users.csv')
movies = pd.read_csv(
    current_path + '/../../../../rllm/datasets/rel-movielens1m/regression/movies.csv')
movies = movies.drop(columns=["Title", "Year", "Runtime", "Plot", "Url"])


## 3. Calculate user average rating and movie average rating
user_rating = {}
for i in user["UserID"].unique():
    user_rating[i] = round(train_df[train_df["UserID"] == i]["Rating"].mean(), 3)

movie_rating = {}
for i in movies["MovielensID"].unique():
    r = round(train_df[train_df["MovieID"] == i]["Rating"].mean(), 3)
    if (not math.isnan(r)):
        movie_rating[i] = r
    else:
        movie_rating[i] = 4

## 4. prepare X and y
rating_movie = np.array(validation_df["MovieID"].replace(movie_rating).fillna(value=4))
rating_user = np.array(validation_df["UserID"].replace(user_rating))
rating_movie_test = np.array(test_df["MovieID"].replace(movie_rating).fillna(value=4))
rating_user_test = np.array(test_df["UserID"].replace(user_rating))
X = np.transpose(np.concatenate(([rating_movie], [rating_user]), axis=0))
X_test = np.transpose(np.concatenate(([rating_movie_test], [rating_user_test]), axis=0))
y = np.transpose(np.array([validation_rating]))
y_test = np.transpose(np.array([test_rating]))

## 5. Train Linear model to preform the task
model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X_test)

## 6. Calculate MAE, running time
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)
t_end = time.time()
print('Total time: ', t_end - t_start)