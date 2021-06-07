# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:26:00 2021

@author: RISHBANS
"""

#import libraries
import pandas as pd

#import data
df = pd.read_csv("pima-data.csv")

#check the correlation
df.corr()
del df['skin']

#Data Molding
diab_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diab_map)

#Store values in X and y
X = df.iloc[:, 0:8]
y = df.iloc[:, 8]


#Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

#Simple Imputer
from sklearn.impute import SimpleImputer
fill_0 = SimpleImputer(missing_values=0, strategy='mean')

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.transform(X_test)

#Naive bayes
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

sfs_c = sfs(nb_model, k_features = 6, forward = True , verbose=3, scoring= 'accuracy', cv=4)

sfs_c = sfs_c.fit(X_train, y_train)
feature_select = list(sfs_c.k_feature_idx_)
print(feature_select)

nb_model.fit(X_train[:, feature_select], y_train)


from sklearn.metrics import accuracy_score
y_pred = nb_model.predict(X_test[:, feature_select])
print(accuracy_score(y_test, y_pred))






























