# tf-idf-clf
# Predicting Movie Genres Based on Plot Summaries
# https://arxiv.org/pdf/1801.04813.pdf
# micro: 0.32867132867132864; macro: 0.13443582645637167
# 0.4622201919555664s
# N/A
# python train.py

import time
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

t_begin = time.time()

# extract useful columns from given tables
col = [0, 1, 3, 9]
train = pd.read_csv("../../../rllm/datasets/rel-movielens1m/classification/movies/train.csv", usecols=col)
val = pd.read_csv("../../../rllm/datasets/rel-movielens1m/classification/movies/validation.csv", usecols=col)
test = pd.read_csv("../../../rllm/datasets/rel-movielens1m/classification/movies/test.csv", usecols=col)

# visualization
# def freq_words(x, terms=30):
#     all_words = ' '.join([text for text in x])
#     all_words = all_words.split()
#     fdist = nltk.FreqDist(all_words)
#     words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})
    
#     d = words_df.nlargest(columns="count", n=terms)
    
#     plt.figure(figsize=(12,15))
#     ax = sns.barplot(data=d, x="count", y="word")
#     ax.set(ylabel='Word')
#     plt.show()


# remove redundant information from plot
def clean_text(text):
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]"," ", text)
    text = ' '.join(text.split())
    text = text.lower()
    return text


# remove meaning words in plot
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


def train_test():
    # train
    # visualization
    # train.head()
    # extract data from 'Genre'
    genres_train = []
    for i in train['Genre']:
        genres_train.append(i.split("|"))

    train['Genre_'] = genres_train
    # train.head()

    genres_val = []
    for i in val['Genre']:
        genres_val.append(i.split("|"))

    val['Genre_'] = genres_val
    # val.head()

    # visualization
    # all_genres = sum(genres_train,[])
    # print(len(set(all_genres)))

    # all_genres = nltk.FreqDist(all_genres)

    # all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 
    #                             'Count': list(all_genres.values())})

    # g = all_genres_df.nlargest(columns="Count", n = 50) 
    # plt.figure(figsize=(12,15)) 
    # ax = sns.barplot(data=g, x= "Count", y = "Genre") 
    # ax.set(ylabel = 'Count') 
    # plt.show()

    # clean plot data
    train['cleanPlot'] = train['Plot'].apply(lambda x: clean_text(str(x)))
    val['cleanPlot'] = val['Plot'].apply(lambda x: clean_text(str(x)))

    # visualization
    # freq_words(train['cleanPlot'], 100)

    # delete meaningless words in the plot
    train['cleanPlot'] = train['cleanPlot'].apply(lambda x: remove_stopwords(x))
    val['cleanPlot'] = val['cleanPlot'].apply(lambda x: remove_stopwords(x))
    
    # visualization
    # freq_words(train['cleanPlot'], 100)

    # label collection
    multilabel_binarizer = MultiLabelBinarizer()
    y = multilabel_binarizer.fit_transform(train['Genre_'])

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

    x_train = train['cleanPlot'].astype(str)
    x_val = val['cleanPlot'].astype(str)

    # apply tf-idf to construct vectorized space
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_val_tfidf = tfidf_vectorizer.transform(x_val)

    # load model
    lr = LogisticRegression()
    clf = OneVsRestClassifier(lr)

    # train and valid
    clf.fit(x_train_tfidf, y)
    y_val = multilabel_binarizer.fit_transform(val['Genre_'])
    clf.fit(x_val_tfidf, y_val)

    # visualization
    # y_pred = clf.predict(x_val_tfidf)
    # multilabel_binarizer.inverse_transform(y_pred)[1]

    # predict on the validation set to fine-tune hyperparameters
    y_pred_prob = clf.predict_proba(x_val_tfidf)
    t = 0.1
    y_pred_new = (y_pred_prob >= t).astype(int)

    # validation result
    f1_score(y_val, y_pred_new, average='micro', zero_division=1)
    f1_score(y_val, y_pred_new, average='macro', zero_division=1)

    # test
    # extract data from 'Genre'
    genres_test = []

    for i in test['Genre']:
        genres_test.append(i.split("|"))

    test['Genre_'] = genres_test

    # clean plot data
    test['cleanPlot'] = test['Plot'].apply(lambda x: clean_text(str(x)))
    test['cleanPlot'] = test['cleanPlot'].apply(lambda x: remove_stopwords(x))

    y_test = multilabel_binarizer.fit_transform(test['Genre_'])
    x_test = test['cleanPlot'].astype(str)

    x_test_tfidf = tfidf_vectorizer.transform(x_test)

    # predict
    y_pred_prob = clf.predict_proba(x_test_tfidf)
    t = 0.095
    q_pred = (y_pred_prob >= t).astype(int)

    print("micro: ", f1_score(y_test, q_pred, average="micro"))
    print("macro: ", f1_score(y_test, q_pred, average="macro"))


if __name__ == "__main__":
    # train & test
    train_test()
    t_end = time.time()
    print("Total time:", t_end - t_begin)