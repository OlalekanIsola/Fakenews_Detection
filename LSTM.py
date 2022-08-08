# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import pickle
import gensim


data = pd.read_csv("dataNew/unbalance5.csv")
data_verif = pd.read_csv("dataNew/validationSet.csv")

y = data["class"].values
y_verif = data_verif["class"].values
X = []
Xverif = []
stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in data["text"].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)

del data

stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in data_verif["text"].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    Xverif.append(tmp)

del data_verif


#initializing word2vec
EMBEDDING_DIM = 100

w2v_model = gensim.models.Word2Vec(sentences=X, vector_size=EMBEDDING_DIM, window=5, min_count=1)



print(w2v_model.wv.most_similar("israel"))

print(w2v_model.wv.most_similar("fbi"))

#tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

with open("tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer, f)

X = tokenizer.texts_to_sequences(X)
Xverif = tokenizer.texts_to_sequences(Xverif)


word_index = tokenizer.word_index
for word, num in word_index.items():
    print(f"{word} -> {num}")
    if num == 10:
        break        
    


maxlen = 700 

#padding to maxlen
X = pad_sequences(X, maxlen=maxlen)
Xverif = pad_sequences(Xverif, maxlen = maxlen)

vocab_size = len(tokenizer.word_index) + 1


def get_weight_matrix(model, vocab):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in vocab.items():
        weight_matrix[i] = model.wv[word]
    return weight_matrix

#using word2vec to generate vectors
embedding_vectors = get_weight_matrix(w2v_model, word_index)

#define the model
model = Sequential()
#use word2vec embedding vectors here
model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=maxlen, trainable=False))
model.add(LSTM(units=128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

del embedding_vectors

model.summary()

X_train, X_test, y_train, y_test = train_test_split(X, y) 

model.fit(X, y, validation_split=0.01, epochs=1)

#prediction

y_pred = (model.predict(Xverif) <= 0.5).astype("int")

accuracy_score(y_verif, y_pred)

print(classification_report(y_verif, y_pred))

print(confusion_matrix(y_verif,y_pred))


y_pred_probas = model.predict(Xverif)
probas = [temp[0] for temp in y_pred_probas]
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot

probasnew = [1.0-temp for temp in probas]
ns_auc = roc_auc_score(y_verif, probasnew)
ns_fpr, ns_tpr, _ = roc_curve(y_verif, probasnew)
pyplot.plot(ns_fpr, ns_tpr, marker='.', label='LSTM')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()
