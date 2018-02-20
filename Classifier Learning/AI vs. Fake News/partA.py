#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:55:45 2017

@author: leyang24kobe
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

categories = ['sci.med', 'rec.autos']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

vectorizer = TfidfVectorizer()
train_vec = vectorizer.fit_transform(newsgroups_train.data)
test_vec = vectorizer.transform(newsgroups_test.data)
        
gnb = MultinomialNB(alpha=.01)
gnb.fit(train_vec, newsgroups_train.target)
pred = gnb.predict(test_vec)

print(metrics.confusion_matrix(pred, newsgroups_test.target))