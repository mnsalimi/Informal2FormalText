import numpy as np
import pandas as pd
import pickle
import gdown
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split #used for splitting data into train and test
from sklearn import metrics  #for scoring performance
from sklearn.model_selection import cross_val_score, StratifiedKFold,KFold,cross_validate
from statistics import mean,variance,stdev

class LogisticRegTokenClassification:
    def __init__(self,splits=25,ngram=3):
        self.__load_model()

        #### TRAINING PARAMETERS ####
        # self.tagged_sentences = []
        # self.__load_data()
        # self.vectorizer = TfidfVectorizer(ngram_range=(1, ngram),max_features=10000)
        # self.logisticRegression = LogisticRegression(C=5e1, solver='lbfgs', random_state=0,max_iter=500)
        # self.skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=25)
        # self.skf = KFold(n_splits=splits, shuffle=True, random_state=0)

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

    def __load_model(self):
        url = "https://drive.google.com/uc?id=1pf4BpJEPfOLcXyBTEeqLwk4r1KWrdLBk"
        output = "./sequence_labeling/logistic_regression/log_reg_token_cls_model.pickle"
        gdown.download(url, output, quiet=False)
        filename = "./log_reg_token_cls_model.pickle" # Modal Path
        loaded_model = pickle.load(open(filename, 'rb'))
        self.logisticRegression = loaded_model["model"]
        self.vectorizer = loaded_model["vectorizer"]

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
                        vector.append("سلام")
                    elif i >= 0 and i < sent_len:
                        vector.append(sent[i][0])
                    else:
                        vector.append("سلام")
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


    def get_cross_validation_scores(self, sequences, labels, scoring_metric='accuracy'):
        return cross_validate(self.logisticRegression, sequences, labels, cv=self.skf, scoring=['accuracy','precision','recall','f1_micro','f1_macro'])
        

    def train_by_cross_validation(self, window_size):
        sequences, labels = self.get_sentences_token_sequences_and_labels(self.tagged_sentences, window_size)
        all_vectors = self.vectorizer.fit_transform(sequences)
        # csv = self.get_cross_validation_scores(all_vectors, labels)
        # return csv
        scores = []
        labels = np.array(labels)
        for train_index, test_index in self.skf.split(all_vectors):
            X_train, X_test = all_vectors[train_index], all_vectors[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            self.logisticRegression.fit(X_train, y_train)
            score = self.logisticRegression.score(X_test, y_test)
            # score = clf.score(X_test, y_test)
            y_predict = self.logisticRegression.predict(X_test)
            score = {"accuracy":0,"recall":0,"percision":0,"f1_macro":0}
            score["accuracy"] = metrics.accuracy_score(y_test,y_predict)
            # score["recall"] = metrics.recall_score(y_test,y_predict,average="macro")
            # score["percision"] = metrics.precision_score(y_test,y_predict,average="macro")
            # score["f1_macro"] = metrics.f1_score(y_test,y_predict,average="macro")
            # score["f1_micro"] = metrics.f1_score(y_test,y_predict,average="micro")
            scores.append(score)
        return scores

    def get_sentence_windows(self, sentence, window_size):
        sent_len = len(sentence)
        start = (-1) * window_size
        end = sent_len + window_size
        sent_vectors = []
        for j in range(start, end - (2 * window_size)):
            vector = []
            for i in range(j, j + (2 * window_size) + 1):
                if i < 0:
                    vector.append("<پد>")
                elif i >= 0 and i < sent_len:
                    vector.append(sentence[i])
                else:
                    vector.append("<پد>")
            sent_vectors.append(self.get_sentence_from_array(vector))
        return sent_vectors

    def train_test_model(self):
        windows, labels = self.get_sentences_token_sequences_and_labels(self.tagged_sentences, 1)
        x_train, x_test, y_train, y_test = self.get_train_test_data(windows, labels)
        self.fit_tfidf_vectors(x_train)
        self.logisticRegression.fit(self.train_vectors, y_train)
        x_test_vectos = self.transform_tfidf_vectors(x_test)
        y_predict = self.logisticRegression.predict(x_test_vectos)
        return metrics.f1_score(y_test, y_predict,average='macro')

    def predict_sentence(self, sentence):
        sentence_sequences = self.get_sentence_windows(sentence.split(" "), 1)
        sequnce_vectors = self.transform_tfidf_vectors(sentence_sequences)
        y_predict = self.logisticRegression.predict(sequnce_vectors)
        return y_predict

if __name__ == "__main__":
    logisticRegression = LogisticRegTokenClassification()

    sentence = input()
    print(logisticRegression.predict_sentence(sentence))

    ##### TRAINING MODEL #####
    # csv = logisticRegression.train_by_cross_validation(1)

    ##### PRINT SCORES #####
    # print(csv)
    # print("mean accuracy: ",mean([score["accuracy"] for score in csv]))
    # print("stdev accuracy: ",stdev([score["accuracy"] for score in csv]))
    
    # print("mean recall: ",mean([score["recall"] for score in csv]))
    # print("stdev recall: ",stdev([score["recall"] for score in csv]))

    # print("mean percision: ",mean([score["percision"] for score in csv]))
    # print("stdev percision: ",stdev([score["percision"] for score in csv]))

    # print("mean f1_micro: ",mean([score["f1_micro"] for score in csv]))
    # print("stdev f1_micro: ",stdev([score["f1_micro"] for score in csv]))

    # print("mean f1_macro: ",mean([score["f1_macro"] for score in csv]))
    # print("stdev f1_macro: ",stdev([score["f1_macro"] for score in csv]))
    
    # print("mean: ", mean_csv)
    # print("variance: ", variance_csv)
    # print("stdev: ", stdev_csv)
    # print("f1: ", logisticRegression.train_test_model())


    ##### save the model to disk #####
    # filename = 'log_reg_token_cls_model.pickle'
    # pickle.dump({"model":logisticRegression.logisticRegression,"vectorizer":logisticRegression.vectorizer}, open(filename, 'wb'))


    ##### PREDICT #####
    # print(logisticRegression.predict_sentence("فک کنم اومدی اون روز"))
