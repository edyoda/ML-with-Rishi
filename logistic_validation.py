# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:01:29 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("pima-data.csv")

#1. Check Correlation
data_cor = dataset.corr()

#2. Delete the correlated feature
del dataset['skin']

#3. Data Molding
diabetes_map = {True: 1, False:0}
dataset['diabetes'] = dataset['diabetes'].map(diabetes_map)

#4. Splitting the data into X and y
X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]

#5. Splitting the X and y into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#6. Imputing
from sklearn.impute import SimpleImputer
s_i = SimpleImputer(missing_values = 0, strategy="mean")
X_train = s_i.fit_transform(X_train)
X_test = s_i.transform(X_test)

#7. apply logictic regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(C = 0.7, max_iter = 200)
lr_model.fit(X_train, y_train)
lr_pred_test = lr_model.predict(X_test)
print(lr_pred_test)

from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits = 4, random_state = 0)
result = cross_val_score(lr_model, X_train, y_train, cv=kfold, scoring='accuracy')
print(result.mean())

from sklearn.model_selection import GridSearchCV
max_iteration = [100, 120, 140, 160, 180]
regularisation = [0.7, 1, 1.5, 2]
param_dict = dict(max_iter = max_iteration, C = regularisation)
lr = LogisticRegression()
grid = GridSearchCV(estimator=lr, param_grid = param_dict, cv = 4)

grid_result = grid.fit(X_train, y_train)
print(grid_result.best_score_, grid_result.best_params_)




















