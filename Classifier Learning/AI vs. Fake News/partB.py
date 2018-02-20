#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:12:15 2017

@author: leyang24kobe
"""

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#Prepare the data
body = pd.read_csv('/Users/leyang24kobe/Desktop/UW/2017 Autumn/CSE 415/HW 7/fnc-1-master/train_bodies.csv')
df = pd.read_csv('/Users/leyang24kobe/Desktop/UW/2017 Autumn/CSE 415/HW 7/fnc-1-master/train_stances.csv')
data = pd.merge(body, df, how='right', on='Body ID')

#Preprocess the data
res_map = {'unrelated': 0, 'discuss': 1, 'agree': 2, 'disagree': 3}
def change(row):
    return res_map[row['Stance']]
data['Stance'] = data.apply(lambda row:change(row), axis = 1)

def cat_text(x):
    res = '%s %s' % (x['Headline'], x['articleBody'])
    return res
data['all_text'] = list(data.apply(cat_text, axis=1))

data_train, data_val = train_test_split(data, test_size = 0.4, random_state = 123)
train_y = data_train['Stance'].values
val_y = data_val['Stance'].values

#Vectorize the data
vectorizer = TfidfVectorizer()
total_train = vectorizer.fit_transform(data['all_text'])
data_train = vectorizer.transform(data_train['all_text'])
data_val = vectorizer.transform(data_val['all_text'])

#Fit XGBoost
dtrain = xgb.DMatrix(data_train, label=train_y)
dval = xgb.DMatrix(data_val, label=val_y)

params_xgb = {
    'max_depth': 6,
    'colsample_bytree': 0.6,
    'subsample': 1.0,
    'eta': 0.1,
    'silent': 1,
    'objective': 'multi:softmax',
    'eval_metric':'mlogloss',
    'num_class': 4
}
evallist = [(dval, 'eval'), (dtrain, 'train')]
num_round = 1000

model = xgb.train(params_xgb, dtrain, num_round, evallist)

#Fit Random Forest
rf = RandomForestClassifier(n_estimators = 25)
rf = rf.fit(data_train, train_y)

#Evaluate XGBoost model
pred_xgb = model.predict(dval)

print("confusion matrix:")
print(metrics.confusion_matrix(val_y,pred_xgb))
print("accuracy score")
print(metrics.accuracy_score(val_y, pred_xgb))

#Evaluate Random Forest model
pred_rf = rf.predict(data_val)

print("confusion matrix:")
print(metrics.confusion_matrix(val_y,pred_rf))
print("accuracy score")
print(metrics.accuracy_score(val_y, pred_rf))


























