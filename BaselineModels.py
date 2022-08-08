# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import gensim
from gensim.models import Word2Vec
import nltk, re, string, collections
from nltk.util import ngrams
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix,roc_curve, auc

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import time
import pickle


""" for TfidfVectorizer"""
def Tfidf(tfid_X_train,y_train,tfid_X_test,y_test):
    
    r_start= time.time()
    #FOR RANDOMFOREST
    randomforest = RandomForestClassifier()
    randomforest.fit(tfid_X_train,y_train)
    with open("RF_TFIDF.pkl","wb") as f:
        pickle.dump(randomforest,f)
    y_pred = randomforest.predict(tfid_X_test)
    score= accuracy_score(y_test,y_pred)
    print("*************************************")
    print(f' TFIDF Random Forest Accuracy: {round(score*100,2)}%')
    R_fpr, R_tpr, threshold = roc_curve(y_test, y_pred)
    R_roc_auc = auc(R_fpr, R_tpr)
    print("This is RandomForest FPR", R_fpr)
    print("This is RandomForest TPR", R_tpr)
    print("This is RandomForest BEST_AUC value", R_roc_auc)
    # cf = confusion_matrix(y_test,y_predict, class=[0,1])
    # print(cf)
    r_stop= time.time()
    total = r_stop-r_start
    print("Randomforest took",total)
    print("*************************************")

    N_start= time.time()
    #FOR Naive Bayesian
    NaiveB = MultinomialNB()
    NaiveB.fit(tfid_X_train,y_train)
    with open("NaiveTFIDF.pkl","wb") as f:
        pickle.dump(NaiveB,f)
    y_pred = NaiveB.predict(tfid_X_test)
    score= accuracy_score(y_test,y_pred)
    print("*************************************")
    print(f'TFIDF Naive Bayes Accuracy: {round(score*100,2)}%')
    N_fpr, N_tpr, threshold = roc_curve(y_test, y_pred)
    N_roc_auc = auc(N_fpr, N_tpr)
    print("This is Naive Bayes FPR", N_fpr)
    print("This is Naive Bayes TPR", N_tpr)
    print("This is Naive Bayes BEST_AUC value", N_roc_auc)
    N_stop= time.time()
    total = N_stop-N_start
    print("Naive Bayes took",total)
    print("*************************************")
    

    S_start= time.time()
    #FOR SVM
    SVM = LinearSVC()
    SVM.fit(tfid_X_train,y_train)
    with open("svmTFIDF.pkl","wb") as f:
        pickle.dump(SVM,f)
    y_pred = SVM.predict(tfid_X_test)
    score= accuracy_score(y_test,y_pred)
    print("*************************************")
    print(f'TFIDF SVM Accuracy: {round(score*100,2)}%')
    S_fpr, S_tpr, threshold = roc_curve(y_test, y_pred)
    S_roc_auc = auc(S_fpr, S_tpr)
    print("This is SVM FPR", S_fpr)
    print("This is SVM TPR", S_tpr)
    print("This is SVM BEST_AUC value", S_roc_auc)
    S_stop= time.time()
    total = S_stop-S_start
    print("SVM took",total)
    print("*************************************")


"""
for bag of words
"""
def BagofWords(BgW_X_train,y_train,BgW_X_test,y_test):
    r_start= time.time()
    #FOR RANDOMFOREST
    randomforest = RandomForestClassifier()
    randomforest.fit(BgW_X_train,y_train)
    with open("RF_BOW.pkl","wb") as f:
        pickle.dump(randomforest,f)
    y_pred = randomforest.predict(BgW_X_test)
    score= accuracy_score(y_test,y_pred)
    print("*************************************")
    print(f'Bag of Words Random Forest Accuracy: {round(score*100,2)}%')
    R_fpr, R_tpr, threshold = roc_curve(y_test, y_pred)
    R_roc_auc = auc(R_fpr, R_tpr)
    print("This is RandomForest FPR", R_fpr)
    print("This is RandomForest TPR", R_tpr)
    print("This is RandomForest BEST_AUC value", R_roc_auc)
    r_stop= time.time()
    total = r_stop-r_start
    print("Randomforest took",total)
    print("*************************************")
    
    N_start= time.time()
    #FOR Naive Bayesian
    NaiveB = MultinomialNB()
    NaiveB.fit(BgW_X_train,y_train)
    with open("NaiveBOW.pkl","wb") as f:
        pickle.dump(NaiveB,f)
    y_pred = NaiveB.predict(BgW_X_test)
    score= accuracy_score(y_test,y_pred)
    print("*************************************")
    print(f'Bag of Words Naive Bayes Accuracy: {round(score*100,2)}%')
    N_fpr, N_tpr, threshold = roc_curve(y_test, y_pred)
    N_roc_auc = auc(N_fpr, N_tpr)
    print("This is Naive Bayes FPR", N_fpr)
    print("This is Naive Bayes TPR", N_tpr)
    print("This is Naive Bayes BEST_AUC value", N_roc_auc)
    N_stop= time.time()
    total = N_stop-N_start
    print("Naive Bayes took",total)
    print("*************************************")
    
    S_start= time.time()
    #FOR SVM
    SVM = LinearSVC()
    SVM.fit(BgW_X_train,y_train)
    with open("svmBOW.pkl","wb") as f:
        pickle.dump(SVM,f)
    y_pred = SVM.predict(BgW_X_test)
    score= accuracy_score(y_test,y_pred)
    print("*************************************")
    print(f'Bag of Words SVM Accuracy: {round(score*100,2)}%')
    S_fpr, S_tpr, threshold = roc_curve(y_test, y_pred)
    S_roc_auc = auc(S_fpr, S_tpr)
    print("This is SVM FPR", S_fpr)
    print("This is SVM TPR", S_tpr)
    print("This is SVM BEST_AUC value", S_roc_auc)
    S_stop= time.time()
    total = S_stop-S_start
    print("SVM took",total)
    print("*************************************")



#data = pd.read_csv("dataNew\\unbalance5.csv")
data = pd.read_csv("data\\Clean.csv")
x = data['text']#independent
y= data['class']#dependent
#split into training and test set
X_train, X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


"""
#TfidfVectorizer
convert text data into a matrix of TF-IDF features
the purpose is to highlight words which are frequent in a document
but not across the document
"""
tfvect = TfidfVectorizer()
#this tokenizes and build vocabularycorpora,models, similarities
tfid_X_train = tfvect.fit_transform(X_train)
with open("TFidfVectorizer.pkl","wb") as f:
    pickle.dump(tfvect,f)
tfid_X_test = tfvect.transform(X_test)
Tfidf(tfid_X_train,y_train,tfid_X_test,y_test)


"""
#BagofWords
Since we are dealing with text data, we cannot fed it directly to our model.
Therefore, I am using bag of words model to extract features from our text
data and convert it into numerical feature vectors that can be fed directly
to the algorithm
"""
vectorizer= CountVectorizer()
#this tokenizes and build vocabulary
BgW_X_train = vectorizer.fit_transform(X_train)
with open("CountVectorizer.pkl","wb") as f:
    pickle.dump(vectorizer,f)
BgW_X_test = vectorizer.transform(X_test)
feature_names= vectorizer.get_feature_names()
BagofWords(BgW_X_train,y_train,BgW_X_test,y_test)


# """
# #Word2Vec
# Word2vec is a two-layer neural net that processes text by “vectorizing” words.
# Its input is a text corpus and its output is a set of vectors: feature vectors
# that represent words in that corpus. While Word2vec is not a deep neural network, it turns text into a numerical form that deep neural networks can understand.
# """
# #Dimensionof vectors we are generating
# #Converting X to format acceptable by gensim, removing annd punctuation stopwords in the process
# X = []
# stop_words = set(nltk.corpus.stopwords.words("english"))
# tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
# for par in data['text'].values:
#     tmp = []
#     sentences = nltk.sent_tokenize(par)
#     for sent in sentences:
#         sent = sent.lower()
#         tokens = tokenizer.tokenize(sent)
#         filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
#         tmp.extend(filtered_words)
#     X.append(tmp)
#
# # model = Word2Vec(X, vector_size=32, window=5, min_count=1)
# #print(model)summarizing the loaded model)
# # model.save('word2vec')
# model=Word2Vec.load('word2vec')
# #see a sample vector for random word, lets say Corona
# # print(model.wv.["corona"])
# # print(modelwv.["trump"])
# p=model.wv.most_similar("fbi")
# print(p)
# #summarize vocabulary
# # words = list(model.wv.vocab)
# # """
#
# #n-Gram
# """
# N-grams represent a continuous sequence of N elements from a given set of texts.
# creates a document term matrix
# The items can be phonemes, syllables, letters, words or base pairs according to
# the application. The n-grams typically are collected from a text or speech corpus
# """
#
# def N_grams(text):
#   """
#   A simple function to clean up the data. All the words that
#   are not designated as a stop word is then lemmatized after
#   encoding and basic regex parsing are performed.
#   """
#   wnl = nltk.stem.WordNetLemmatizer()
#   stopwords = nltk.corpus.stopwords.words('english')
#   text = (unicodedata.normalize('NFKD', text)
#     .encode('ascii', 'ignore')
#     .decode('utf-8', 'ignore')
#     .lower())
#   words = re.sub(r'[^\w\s]', '', text).split()
#   return [wnl.lemmatize(word) for word in words if word not in stopwords]
#
# words = N_grams(''.join(str(data['text'].tolist())))
# # implementing bi-gram for dataset
# bigram_all=(pd.Series(nltk.ngrams(words, 2)).value_counts())[:30]
# bigram_all=pd.DataFrame(bigram_all)
# print(bigram_all)
#
# # len(vect.get_feature_names())
#

"""
For Second part
"""

def all_together(X_train,y_train,X_test,y_test):
    r_start= time.time()
    #FOR RANDOMFOREST
    randomforest = RandomForestClassifier()
    randomforest.fit(X_train,y_train)
    y_pred = randomforest.predict(X_test)
    score= accuracy_score(y_test,y_pred)
    print("*************************************")
    print("This is for the second part")
    print(f'Bag of Words Random Forest Accuracy: {round(score*100,2)}%')
    R_fpr, R_tpr, threshold = roc_curve(y_test, y_pred)
    R_roc_auc = auc(R_fpr, R_tpr)
    print("This is RandomForest FPR", R_fpr)
    print("This is RandomForest TPR", R_tpr)
    print("This is RandomForest BEST_AUC value", R_roc_auc)
    r_stop= time.time()
    total = r_stop-r_start
    print("Randomforest took",total)
    print("*************************************")


"""
#BagofWords
Since we are dealing with text data, we cannot fed it directly to our model.
Therefore, I am using bag of words model to extract features from our text
data and convert it into numerical feature vectors that can be fed directly
to the algorithm
"""
vectorizer= CountVectorizer()
#this tokenizes and build vocabulary
BgW_X_train = vectorizer.fit_transform(X_train)
BgW_X_test = vectorizer.transform(X_test)
feature_names= vectorizer.get_feature_names()
BagOfWords = all_together(BgW_X_train,y_train,BgW_X_test,y_test)
print("This is for bag of words")




"""
#TfidfVectorizer
convert text data into a matrix of TF-IDF features
the purpose is to highlight words which are frequent in a document
but not across the document
"""
tfvect = TfidfVectorizer()
#this tokenizes and build vocabularycorpora,models, similarities
tfid_X_train = tfvect.fit_transform(X_train)
tfid_X_test = tfvect.transform(X_test)
tfVect = all_together(tfid_X_train,y_train,tfid_X_test,y_test)
print("This is for TFdif")
