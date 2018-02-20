#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 14:55:35 2017

@author: leyang24kobe
"""
#amount_tsh - Total static head (amount water available to waterpoint)
#date_recorded - The date the row was entered
#funder - Who funded the well
#gps_height - Altitude of the well
#installer - Organization that installed the well
#longitude - GPS coordinate
#latitude - GPS coordinate
#wpt_name - Name of the waterpoint if there is one
#num_private -
#basin - Geographic water basin
#subvillage - Geographic location
#region - Geographic location
#region_code - Geographic location (coded)
#district_code - Geographic location (coded)
#lga - Geographic location
#ward - Geographic location
#population - Population around the well
#public_meeting - True/False
#recorded_by - Group entering this row of data
#scheme_management - Who operates the waterpoint
#scheme_name - Who operates the waterpoint
#permit - If the waterpoint is permitted
#construction_year - Year the waterpoint was constructed
#extraction_type - The kind of extraction the waterpoint uses
#extraction_type_group - The kind of extraction the waterpoint uses
#extraction_type_class - The kind of extraction the waterpoint uses
#management - How the waterpoint is managed
#management_group - How the waterpoint is managed
#payment - What the water costs
#payment_type - What the water costs
#water_quality - The quality of the water
#quality_group - The quality of the water
#quantity - The quantity of water
#quantity_group - The quantity of water
#source - The source of the water
#source_type - The source of the water
#source_class - The source of the water
#waterpoint_type - The kind of waterpoint
#waterpoint_type_group - The kind of waterpoint


import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import tree
import seaborn
import matplotlib.pyplot as plt



x_train = pd.read_csv('Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_values.csv')
y_train = pd.read_csv('Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_labels.csv')
x_test = pd.read_csv('Pump_it_Up_Data_Mining_the_Water_Table_-_Test_set_values.csv')

x_train = x_train[['id', 'amount_tsh', 'gps_height', 'longitude', 'latitude', 'region_code', 'district_code', 
                  'population', 'public_meeting', 'construction_year']]



#training set
data = pd.concat([x_train, y_train[['status_group']]], axis = 1)
data = data.dropna()
data = data[data.construction_year != 0]



mapping_sg = {"functional":2, "functional needs repair":1, "non functional":0}


def map_pm(row):
    if row['public_meeting'] == True:
        return 1
    else:
        return 0

data = data.replace({"status_group":mapping_sg})
data['public_meeting'] = data.apply(lambda row: map_pm(row), axis = 1)

def num_year(row):
    return 2017 - row['construction_year']

data['construction_year'] = data.apply(lambda row:num_year(row), axis = 1)


#Plot
data.iloc[:, range(1,11)].describe()

seaborn.countplot(x = 'status_group', data = y_train)
plt.title('Mine Status')
plt.xlabel('Status')
plt.ylabel('Frequency')


fig1 = seaborn.PairGrid(data, y_vars=['status_group'], x_vars=['amount_tsh', 'gps_height'], palette="GnBu_d")
fig1.map(plt.scatter, s=50, edgecolor="white")
fig1 = seaborn.PairGrid(data, y_vars=['status_group'], x_vars=['longitude', 'latitude'], palette="GnBu_d")
fig1.map(plt.scatter, s=50, edgecolor="white")
fig1 = seaborn.PairGrid(data, y_vars=['status_group'], x_vars=['region_code', 'district_code'], palette="GnBu_d")
fig1.map(plt.scatter, s=50, edgecolor="white")
fig1 = seaborn.PairGrid(data, y_vars=['status_group'], x_vars=['population', 'construction_year'], palette="GnBu_d")
fig1.map(plt.scatter, s=50, edgecolor="white")


plt.title('Figure 1. Association Between Quantitative Predictors and Mine Status', 
                    fontsize = 12, loc='right')
fig1.savefig('reportfig1.jpg')

box1 = seaborn.boxplot(x='public_meeting', y="status_group", data=data)
box1 = plt.gcf()
plt.title("Figure 2. Association Between Qualitative Predictors and Mine Status", fontsize = 12, loc='right')
box1.savefig('reportfig2.jpg')



#change test set
x_test = x_test[['id', 'amount_tsh', 'gps_height', 'longitude', 'latitude', 'region_code', 'district_code', 
                  'population', 'public_meeting', 'construction_year']]
x_test = x_test.dropna()
x_test = x_test[x_test.construction_year != 0]

x_test['public_meeting'] = x_test.apply(lambda row: map_pm(row), axis = 1)
x_test['construction_year'] = x_test.apply(lambda row: num_year(row), axis = 1)

train, cv = train_test_split(data, test_size = 0.4, random_state = 123)

train_pred = train.iloc[:, range(1,10)]
train_res = train['status_group']
cv_pred = cv.iloc[:, range(1, 10)]
cv_res = cv['status_group']

#decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_pred, train_res)
dt_pred = clf.predict(cv_pred)

print("confusion matrix:")
print(metrics.confusion_matrix(cv_res,dt_pred))
print("accuracy score")
print(metrics.accuracy_score(cv_res, dt_pred))

#random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 25)
rf = rf.fit(train_pred, train_res)
rf_pred = rf.predict(cv_pred)

print("confusion matrix:")
print(metrics.confusion_matrix(cv_res,rf_pred))
print("accuracy score")
print(metrics.accuracy_score(cv_res, rf_pred))

#adaboost
from sklearn.ensemble import AdaBoostClassifier
ab = AdaBoostClassifier()
ab = ab.fit(train_pred, train_res)
ab_pred = ab.predict(cv_pred)

print("confusion matrix:")
print(metrics.confusion_matrix(cv_res,ab_pred))
print("accuracy score")
print(metrics.accuracy_score(cv_res, ab_pred))

#Final assessment
x_final = data.iloc[:, range(1,10)]
y_final = data['status_group']
rf_final = RandomForestClassifier(n_estimators = 25)
rf_final = rf_final.fit(x_final, y_final)
rf_final_pred = rf_final.predict(x_final)

print("confusion matrix:")
print(metrics.confusion_matrix(y_final,rf_final_pred))
print("accuracy score")
print(metrics.accuracy_score(y_final, rf_final_pred))


from sklearn.decomposition import PCA 
pca_fit = PCA(n_components = 2)
pca_fit.fit(cv_pred)
pca_trans = pca_fit.transform(cv_pred)

#plot result
plt.scatter(x=pca_trans[:,0], y=pca_trans[:,1], c=clf.predict(cv_pred))
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 2 Clusters')
plt.show()
