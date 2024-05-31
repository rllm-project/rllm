# lm embedding for classification task in rel-movielens1M
# Paper: 
# f1_micro:  0.4706292892869606  f1_macro:  0.3306643865453048
# Cost: $2.58e-08
# Runtime: 16.3s (fastest 15.8s slowest 17.6s on my laptop, mainly depend on how hot your computer is)
# Description: Use lm embedding to do classification
# Usage: python

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import time
from sklearn import svm
from sentence_transformers import SentenceTransformer
from utils import get_lm_emb_cost

## 1. Start clock
t_start = time.time()

## 2. Load Data
current_path = 'F:\\rllm2\\examples\\group_9_method\\rel-movielenslm\\classification'
test_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/classification/movies/test.csv')
train_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/classification/movies/train.csv')
validation_df = pd.read_csv(
	current_path + '/../../../../rllm/datasets/rel-movielens1m/classification/movies/validation.csv')
train_val = pd.concat([train_df, validation_df])
train_val["Set"] = 0
test_df["Set"] = 1
movie_all = pd.concat([test_df, train_val]).reset_index()


## 3. Embed data (title and plot)
model = SentenceTransformer(current_path + '/../all-MiniLM-L6-v2')

sentences = (movie_all["Title"]+" " +movie_all["Plot"]).fillna(value="").tolist()
embeddings = model.encode(sentences)


## 4. Data preprocess to extract other binary features
movie_directors = movie_all["Director"].str.get_dummies(sep=',')
movie_rate =  movie_all["Certificate"].str.get_dummies()

movie = pd.concat([movie_directors, movie_rate, movie_all[["Set", "Genre"]]], axis=1).join(pd.DataFrame(embeddings))

## 5. Prepare X and y
train = movie[movie["Set"]==0]
test = movie[movie["Set"]==1]

train_genres = train['Genre'].str.get_dummies(sep='|')
train =train.drop(columns=["Genre", "Set"])

test_genres =  test['Genre'].str.get_dummies(sep='|')
test = test.drop(columns=["Genre", "Set"])


## 6. Train 18 linear SVM for the task
X = np.array(train)
X_test = np.array(test)
pred = []
C = [1.2, 1.5, 1.4, 1.3, 1.4, 1.5, 
     1.4, 1.1, 1.3, 1.1, 1.1, 1.4,
       1, 1.1, 1.4, 1.4, 1.6, 1.5]
# if this is too much parameters,
# replace C[j] in the following sector with 1.4, the result is still decent 0.3979
j = 0
for column in train_genres.columns:
    y = np.array(train_genres[column])
    # y_test = test_genres[column]
    model = svm.SVC(kernel='linear')
    model.fit(X, y)
    
    predictions = model.predict(X_test)
    pred.append(predictions)
    j += 1

pred = np.array(pred).transpose()

## 7. calculate metrics, running time and cost
f1_score_micro = f1_score(np.array(test_genres), pred, average='micro')
f1_score_macro = f1_score(np.array(test_genres), pred, average='macro')

print("f1_micro: ", f1_score_micro, " f1_macro: ", f1_score_macro)
t_end = time.time()
print('Total time: ', t_end - t_start)

total_cost = get_lm_emb_cost(''.join(sentences))
print('Total cost: ', total_cost)