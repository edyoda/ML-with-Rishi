# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 20:45:58 2021

@author: RISHBANS
"""

import pandas as pd

dataset = pd.read_csv('BankNote_Authentication.csv')

X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

from sklearn.ensemble import GradientBoostingClassifier
gb_c = GradientBoostingClassifier(n_estimators=100, random_state = 0, max_depth = 50, learning_rate = 0.01)
gb_c.fit(X_train, y_train)

print(gb_c.score(X_test, y_test))


from sklearn.model_selection import GridSearchCV
n_est = [100, 110, 120, 150]
max_d = [25, 35, 45, 55, 65]
lr = [ 0.01, 0.05, 0.1]

param_grid = dict(n_estimators = n_est, max_depth = max_d, learning_rate = lr)

gb = GradientBoostingClassifier()
grid = GridSearchCV(estimator = gb, param_grid = param_grid, cv = 4, n_jobs = -1)

grid_result = grid.fit(X_train, y_train)

print(grid_result.best_params_)
