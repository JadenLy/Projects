#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:20:21 2017

@author: leyang24kobe
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

#Load data
df = pd.read_csv('pollution.csv')


#a: perform PCA
pca_test = PCA(n_components = 16)
pca_test.fit(df)

#Plot the Proportion of explained variance with respect to the number of predictors. 
plt.plot(np.cumsum(pca_test.explained_variance_ratio_ * 100))


#b: K-mean clustering
clusters = range(1, 10)
meandist = []

#Calculate the average distances between each observation to its labelled centroid. 
for k in clusters:
    model = KMeans(n_clusters = k)
    model.fit(df)
    assign = model.predict(df)
    meandist.append(sum(np.min(cdist(df, model.cluster_centers_, 'euclidean'), axis=1)) 
    / df.shape[0])

#Plot the average distance with respect to the number of clusters. 
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

#Fit KMeans with clusters
km_model = KMeans(n_clusters = 2)
km_model.fit(df)
clusassign = km_model.predict(df)

#Perform PCA with 2 components for plot. 
pca_fit = PCA(n_components = 2)
pca_fit.fit(df)
pca_trans = pca_fit.transform(df)

#plot clustering result
plt.scatter(x=pca_trans[:,0], y=pca_trans[:,1], c=km_model.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 2 Clusters')
plt.show()

#c: Hierarchial clustering
dist_hc = []
hc_model = AgglomerativeClustering(n_clusters = 2)
assign = hc_model.fit_predict(df)

#Plot the clustering result
plt.scatter(x=pca_trans[:,0], y=pca_trans[:,1], c=hc_model.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 2 Clusters')
plt.show()

#d: linear model
#Separate into train and cv sets
df_x = df.drop(' SO', axis = 1)
df_y = df[' SO']
train, cv = train_test_split(df, test_size = 0.4, random_state = 123)
x_train = train.drop(' SO', axis = 1)
y_train = train[' SO']
x_cv = cv.drop(' SO', axis = 1)
y_cv = cv[' SO']

#Least Squares
lm_fit = LinearRegression()
lm_fit.fit(x_train, y_train)

np.mean((lm_fit.predict(x_cv) - y_cv)**2)
#2017.2242086892982


#Ridge
alpha = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}
ridge_fit = GridSearchCV(Ridge(normalize = True), alpha, cv = 10)
ridge_fit.fit(df_x, df_y)
ridge_fit.best_params_
#{'alpha': 1}

best_rd_fit = Ridge(normalize = True)
best_rd_fit.fit(x_train, y_train)

np.mean((best_rd_fit.predict(x_cv) - y_cv)**2)
#3999.622853320428

#Lasso
lasso_fit = GridSearchCV(Lasso(normalize = True), alpha, cv = 10)
lasso_fit.fit(df_x, df_y)
lasso_fit.best_params_
#{'alpha': 1}

best_la_fit = Lasso(normalize = True)
best_la_fit.fit(x_train, y_train)

np.mean((best_la_fit.predict(x_cv) - y_cv)**2)
#3844.0250942849466

#e: Classification
#Set up data
so_med = np.median(df[' SO'])
data = df
def categorize(row):
    if (row[' SO'] >= so_med):
        return 1
    else :
        return 0

data[' SO'] = data.apply(lambda row: categorize(row), axis = 1)

#Split into training and cross-validation set. 
data_x = data.drop(' SO', axis = 1)
data_y = data[' SO']
train_clas, cv_clas = train_test_split(df, test_size = 0.4, random_state = 123)
x_clas_train = train_clas.drop(' SO', axis = 1)
y_clas_train = train_clas[' SO']
x_clas_cv = cv_clas.drop(' SO', axis = 1)
y_clas_cv = cv_clas[' SO']

#SVM
#Search for the best combination of parameters
tuning_param = [{'C': [0.01, 0.1, 1, 5, 10, 100],
                 'gamma': [0.01, 0.1, 1, 5, 10, 100]}]
    
svm_fit = GridSearchCV(SVC(kernel = 'rbf'), tuning_param, cv = 10)
svm_fit.fit(data_x, data_y)

svm_fit.best_params_
#{'C': 0.01, 'gamma': 0.1}

#Fit the model using the parameters found
svm_best_fit = SVC(kernel = 'rbf', C = 0.01, gamma = 0.1)
svm_best_fit.fit(x_clas_train, y_clas_train)
np.mean(svm_best_fit.predict(x_clas_cv) - y_clas_cv)
#0.5833333333333334

#LDA
lda_fit = lda()
lda_fit.fit(x_clas_train, y_clas_train)

np.mean(lda_fit.predict(x_clas_cv) - y_clas_cv)
#0.20833333333333334


#QDA
qda_fit = qda()
qda_fit.fit(x_clas_train, y_clas_train)

np.mean(qda_fit.predict(x_clas_cv) - y_clas_cv)
#0.4583333333333333







































