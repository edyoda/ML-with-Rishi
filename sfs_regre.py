# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 20:30:14 2021

@author: RISHBANS
"""

#import libraries
import pandas as pd

#import data
df = pd.read_csv("Largecap_Balancesheet.csv")


#Store values in X and y
X = df.iloc[:, 0:4]
y = df.iloc[:, -1]


#Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.transform(X_test)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

sfs_r = sfs(linear_model, k_features = 3, forward = True , verbose=3, scoring= 'r2', cv=4)

sfs_r = sfs_r.fit(X_train, y_train)
feature_select = list(sfs_r.k_feature_idx_)

print(feature_select)


linear_model.fit(X_train[:, feature_select], y_train)

y_pred = linear_model.predict(X_test[:, feature_select])

print(y_pred)




















