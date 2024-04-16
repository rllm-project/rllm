# tf-idf-clf
# Predicting Movie Genres Based on Plot Summaries
# https://arxiv.org/pdf/1801.04813.pdf
# micro: 0.3643202579258463; macro: 0.06953107551293945
# 12.427888631820679s
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
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

t_begin = time.time()

# extract useful columns from given tables
col = [0, 2, 6]
meta = pd.read_csv("../dataset/wiki_dataset.csv", usecols=col)

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
    no_stopword_text = [w for w in text.split() if w not in stop_words]
    return ' '.join(no_stopword_text)


def train_test():
    # train
    # visualization
    # meta.head()
    # extract data from 'Genre'
    genres = []
    for i in meta['Genre']:
        genres.append(i.split("|"))

    meta['Genre_'] = genres
    # meta.head()

    # visualization
    # all_genres = sum(genres,[])
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
    meta['cleanPlot'] = meta['Plot'].apply(lambda x: clean_text(str(x)))

    # visualization
    # freq_words(meta['cleanPlot'], 100)

    # delete meaningless words in the plot
    meta['cleanPlot'] = meta['cleanPlot'].apply(lambda x: remove_stopwords(x))
    
    # visualization
    # freq_words(meta['cleanPlot'], 100)

    # label collection
    multilabel_binarizer = MultiLabelBinarizer()
    y = multilabel_binarizer.fit_transform(meta['Genre_'])

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

    x_train, x_test, y_train, y_test = train_test_split(meta['cleanPlot'], y, test_size=0.2, random_state=9)
    # x_train, x_test, y_train, y_test = train_test_split(meta['cleanPlot'], y, test_size=0.2, stratify=meta['Genre_'])

    # apply tf-idf to construct vectorized space
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)

    # load model
    lr = LogisticRegression()
    clf = OneVsRestClassifier(lr)

    # train and valid
    clf.fit(x_train_tfidf, y_train)

    # visualization
    # y_pred = clf.predict(x_test_tfidf)
    # multilabel_binarizer.inverse_transform(y_pred)[1]

    # test
    # predict
    y_pred_prob = clf.predict_proba(x_test_tfidf)
    t = 0.4
    q_pred = (y_pred_prob >= t).astype(int)

    print("micro: ", f1_score(y_test, q_pred, average="micro"))
    print("macro: ", f1_score(y_test, q_pred, average="macro"))


if __name__ == "__main__":
    # train & test
    train_test()
    t_end = time.time()
    print("Total time:", t_end - t_begin)
