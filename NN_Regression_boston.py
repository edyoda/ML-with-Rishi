# -*- coding: utf-8 -*-
"""
Created on Thu May  6 20:18:21 2021

@author: RISHBANS
"""

from sklearn.datasets import load_boston
import pandas as pd
boston = load_boston()
df = pd.DataFrame(boston.data)

df['House_Price'] = boston.target

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.neighbors import KNeighborsRegressor
NN_model = KNeighborsRegressor(n_neighbors = 3)
NN_model.fit(X_train, y_train)

y_predict = NN_model.predict(X_test)
print(y_predict)
print(NN_model.score(X_train, y_train))
print(NN_model.score(X_test, y_test))

