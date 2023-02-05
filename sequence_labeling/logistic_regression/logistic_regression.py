import numpy as np
import pandas as pd
import math
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split #used for splitting data into train and test
from sklearn import metrics  #for scoring performance
from sklearn.model_selection import cross_val_score, StratifiedKFold
from statistics import mean

class LogisticRegTokenClassification:
    def __init__(self):
        self.tagged_sentences = []
        self.__load_data()
        self.vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 3),max_features=200000)
        self.logisticRegression = LogisticRegression(C=5e1, solver='lbfgs', multi_class='multinomial', random_state=41,n_jobs=-1)
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)

    def __load_data(self):
        df1 = pd.read_csv("../data/Data - Sheet1.csv")
        df2 = pd.read_csv("../data/Data - Sheet2.csv")
        df3 = pd.read_csv("../data/Data - Sheet3.csv")
        all_df = pd.concat([df1, df2, df3], axis=0)
        tokens = np.asarray(all_df)
        tagged_sentence = []
        for token in tokens:
            if pd.isna(token[0]):
                self.tagged_sentences.append(tagged_sentence)
                tagged_sentence = []
            else:
                tagged_sentence.append((token[0], token[1]))
        # print(self.tagged_sentences[0])

    def get_sentence_from_array(self,sent_array):
        return " ".join(sent_array)

    def get_sentences_token_sequences_and_labels(self, sentences, window_size, save_labels=True):
        sent_vectors = []
        sent_vectors_lables = []
        for sent in sentences:
            sent_len = len(sent)
            start = (-1) * window_size
            end = sent_len + window_size
            for j in range(start, end - (2 * window_size)):
                vector = []
                for i in range(j, j + (2 * window_size) + 1):
                    if i < 0:
                        vector.append("<pad>")
                    elif i >= 0 and i < sent_len:
                        vector.append(sent[i][0])
                    else:
                        vector.append("<pad>")
                sent_vectors.append(self.get_sentence_from_array(vector))
                if save_labels:
                    sent_vectors_lables.append(sent[j + window_size][1])
        return sent_vectors, sent_vectors_lables;

    def get_train_test_data(self, sequences, labels, test_size=0.2, random_state=25):
        x_train, x_test, y_train, y_test = train_test_split(sequences, labels, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test;

    def fit_tfidf_vectors(self, sequences):
        self.train_vectors = self.vectorizer.fit_transform(sequences)

    def transform_tfidf_vectors(self, sequences):
        return self.vectorizer.transform(sequences)


    def get_cross_validation_scores(self, sequences, labels, scoring_metric='f1_macro'):
        return cross_val_score(self.logisticRegression, sequences, labels, cv=self.skf, scoring=scoring_metric)

    def train_by_cross_validation(self, window_size):
        sequences, labels = self.get_sentences_token_sequences_and_labels(self.tagged_sentences, window_size)
        all_vectors = self.vectorizer.fit_transform(sequences)
        csv = self.get_cross_validation_scores(all_vectors, labels)
        return csv

    def get_sentence_windows(self, sentence, window_size):
        sent_len = len(sentence)
        start = (-1) * window_size
        end = sent_len + window_size
        sent_vectors = []
        for j in range(start, end - (2 * window_size)):
            vector = []
            for i in range(j, j + (2 * window_size) + 1):
                if i < 0:
                    vector.append("<pad>")
                elif i >= 0 and i < sent_len:
                    vector.append(sentence[i])
                else:
                    vector.append("<pad>")
            sent_vectors.append(self.get_sentence_from_array(vector))
        return sent_vectors

    def predict_sentence(self, sentence):
        sentence_sequences = self.get_sentence_windows(sentence.split(" "), 1)
        sequnce_vectors = self.transform_tfidf_vectors(sentence_sequences)
        y_predicts = self.logisticRegression.predict(sequnce_vectors)
        return y_predicts

    def train_model(self):
        windows, labels = self.get_sentences_token_sequences_and_labels(self.tagged_sentences, 1)
        x_train, x_test, y_train, y_test = self.get_train_test_data(windows, labels)
        self.fit_tfidf_vectors(x_train)
        self.logisticRegression.fit(self.train_vectors, y_train)


if __name__ == "__main__":
    logisticRegression = LogisticRegTokenClassification()
    # windows, labels = logisticRegression.get_sentences_windows_labels(logisticRegression.tagged_sentences,0)
    # mean_csv = mean(logisticRegression.train_by_cross_validation(1))
    # print("mean: ", mean_csv)
    logisticRegression.train_model()
    print(logisticRegression.predict_sentence("سلام باشه اومدم"))
