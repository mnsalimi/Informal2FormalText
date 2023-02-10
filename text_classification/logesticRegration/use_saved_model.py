import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import  TfidfVectorizer
import pickle


vectorizer =TfidfVectorizer()
symboles = [',','.','!','؟','(',')','<','>']   

def normalize(text):
    text = re.sub(r'[^\u0621-\u06CC\s]+', '', text)
    words = text.split(" ")
    words_without_stopwords = [word for word in words if not word in symboles]
    return " ".join(words_without_stopwords)


def prediction(clf,sentece):
    sentence=normalize(sentece)
    sentence = clf["tfidf"].transform([sentece])
    formality = clf['clf'].predict(sentence)[0]
    print(formality)

with open('logreg_trained_model12.pickle', 'rb') as file:
    model = pickle.load(file)
new_sentence=input()
prediction(model,new_sentence)