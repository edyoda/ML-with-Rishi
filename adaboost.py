# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 19:24:19 2021

@author: RISHBANS
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

adaboost = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=100), n_estimators=100)

adaboost.fit(X_train, y_train)
print(adaboost.score(X_test, y_test))


#Bagging
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))


#Adaboost with Logistic Regression
from sklearn.linear_model import LogisticRegression
ada_lr = AdaBoostClassifier(base_estimator=LogisticRegression(max_iter=150), n_estimators=100)
ada_lr.fit(X_train, y_train)
print(ada_lr.score(X_test, y_test))

y_pred = ada_lr.predict(X_test)
from sklearn.metrics import f1_score
print(f1_score(y_pred, y_test, average='macro'))

