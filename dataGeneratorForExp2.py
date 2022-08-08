# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
#from FRP_TPR_AUC import Tfidf
#from FRP_TPR_AUC import BagofWords
#from sklearn.feature_extraction.text import TfidfVectorizer



#fake_news.head()
#true_news.head()


def pick_up_clear_data():
    fake_news = pd.read_csv("data\\Fake.csv", index_col=False)
    fake_news["text"] = fake_news["text"] + " " + fake_news["title"]
    negetive = pd.DataFrame(fake_news['text'])
    neg_ver = negetive.sample(frac=0.30)
    neg_train = negetive.drop(neg_ver.index)

    true_news = pd.read_csv("data\\True.csv", index_col=False)
    true_news["text"] = true_news["text"] + " " + true_news["title"]
    positve = pd.DataFrame(true_news['text'])
    pos_ver = positve.sample(frac=0.30)
    pos_train = positve.drop(pos_ver.index)

    neg_ver['class'] = 0
    pos_ver['class'] = 1

    clean_ver = pd.concat([neg_ver, pos_ver])
    clean_ver = clean_ver.sample(frac=1)

    return clean_ver, neg_train, pos_train


def generate_robustness_data (neg_train, pos_train, FP_rate, FN_rate ):

    FN=neg_train.sample(frac=FN_rate)
    TN=neg_train.drop(FN.index)
    FN['class']=0
    TN['class']=1

    rob_neg = pd.concat([FN, TN])

    FP = pos_train.sample(frac=FP_rate)
    TP = pos_train.drop(FP.index)
    FP['class'] = 1
    TP['class'] = 0

    rob_pos = pd.concat([FP, TP])

    rob_train=pd.concat([rob_neg,rob_pos])
    rob_train=rob_train.sample(frac=1)

    return rob_train


clean_ver,neg_train,pos_train=pick_up_clear_data()

clean_ver.to_csv("dataNew\\clean_for_verify.csv")


def generate_imbalanced_data (neg_train,pos_train, percOfNeg):
    
    undersampleNeg = neg_train.sample(frac = percOfNeg)
    undersampleNeg = neg_train.drop(undersampleNeg.index)
    undersampleNeg['class'] = 1
    print(len(undersampleNeg))
    #print(len())
    
    pos_train['class'] = 0
    totalData = pd.concat([undersampleNeg,pos_train])
    totalData = totalData.sample(frac = 1)
    return totalData
    



rob_train=pd.DataFrame()

for miss_rate in range(5,31,5):

    rob_train.drop(rob_train.index,inplace=True)

    rob_train=generate_robustness_data (neg_train, pos_train, miss_rate/100, miss_rate/100)

    trainfile="dataNew\\mislabled"+str(miss_rate)+".csv"

    rob_train.to_csv(trainfile)

unbalance_rate=[5,25,50,75,95]

for rate in unbalance_rate:

    rob_train.drop(rob_train.index,inplace=True)

    rob_train=generate_imbalanced_data (neg_train, pos_train, rate/100)

    trainfile="dataNew\\unbalance"+str(rate)+".csv"

    rob_train.to_csv(trainfile)