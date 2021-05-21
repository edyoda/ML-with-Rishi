# -*- coding: utf-8 -*-
"""
Created on Fri May 21 19:55:22 2021

@author: RISHBANS
"""

import pandas as pd 
import numpy as np
#import the dataset
dataset = pd.read_csv("HR.csv")

#Store features and target variables in X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 7].values

#Split into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#apply standard scaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
trainX = sc_X.fit_transform(X_train)
testX = sc_X.transform(X_test)

#apply logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1, max_iter=150)
lr.fit(trainX, y_train)
y_pred = lr.predict(testX)
print(lr.score(testX, y_test))

#Print the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Apply K-Fold
from sklearn.model_selection import KFold, cross_val_score
kfold = KFold(n_splits = 4, random_state = 0)
result = cross_val_score(lr, trainX, y_train, cv=kfold, scoring='accuracy')
print(result)

#Apply Grid Search
from sklearn.model_selection import GridSearchCV
max_iter = [100, 110, 120, 130, 140, 150]
C = [0.7, 1, 1.2, 1.4]

param_grid = dict(max_iter=max_iter, C=C)
lr_gs = LogisticRegression()
grid = GridSearchCV(estimator = lr_gs, param_grid=param_grid, cv = 4)
grid_result = grid.fit(trainX, y_train)
print(grid_result.best_score_, grid_result.best_params_)



#Pipeline
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

digit_pipeline = make_pipeline(StandardScaler(), LogisticRegression())

digit_pipeline.fit(X_train, y_train)
y_pred = digit_pipeline.predict(X_test)
print(accuracy_score(y_test, y_pred))

from sklearn.feature_selection import SelectKBest, f_classif

digit_pipeline = make_pipeline(StandardScaler(), SelectKBest(k=3, score_func=f_classif)
                               ,LogisticRegression(C=1, max_iter=150))

digit_pipeline.fit(X_train, y_train)
y_pred = digit_pipeline.predict(X_test)
print(accuracy_score(y_test, y_pred))


#Apply Gridsearch to find selectkbest, C, max_iter
params = {'selectkbest__k': [1, 2, 3,4,5,6],
          'logisticregression__C': [0.7, 1, 1.2, 1.5],
          'logisticregression__max_iter' : [100, 120, 140, 160]          
            }

grid = GridSearchCV(digit_pipeline, param_grid=params, cv = 4)
grid_result = grid.fit(trainX, y_train)
print(grid_result.best_score_, grid_result.best_params_)











