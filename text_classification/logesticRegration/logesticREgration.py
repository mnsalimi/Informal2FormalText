import numpy as np
import pandas as pd
import re
import math
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split #used for splitting data into train and test
from sklearn import metrics  #for scoring performance
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from statistics import mean
import stopwordsiso as stopwords


persian_stopwords = stopwords.stopwords("fa")
vectorizer =TfidfVectorizer()
kfold = KFold(n_splits=20, shuffle=True, random_state=0)
symboles = [',','.','!','؟','(',')','<','>']   

def normalize(text):
    text = re.sub(r'[^\u0621-\u06CC\s]+', '', text)
    words = text.split(" ")
    words_without_stopwords = [word for word in words if not word in symboles]
    return " ".join(words_without_stopwords)

def remove_stop_Word(text):
    words = text.split(" ")
    words_without_stopwords = [word for word in words if not word in persian_stopwords]
    return " ".join(words_without_stopwords)


def train_by_cross_validation(X,y):
    X=vectorizer.fit_transform(X)
    clf = LogisticRegression()
    scores = []
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
    return clf,score

def prediction(clf,sentece):
    sentence=normalize(sentece)
    sentence = vectorizer.transform([sentece])
    formality = clf.predict(sentence)[0]
    print(formality)


df = pd.read_csv("classification.csv")
df['sentence'] = df['sentence'].apply(lambda x: normalize(x))
sentences_whithout_stop_word=df['sentence'].apply(lambda x:remove_stop_Word(x))
sentences=df['sentence']
label=df['label']
clf,scores=train_by_cross_validation(sentences,label)
clf2,scores2=train_by_cross_validation(sentences_whithout_stop_word,label)
mean_accuracy = np.mean(scores)
mean_accuracy2 = np.mean(scores2)
print("Mean accuracy:", mean_accuracy)
print("Mean accuracy by removing stop words:", mean_accuracy2)
new_sentence=input()
prediction(new_sentence)