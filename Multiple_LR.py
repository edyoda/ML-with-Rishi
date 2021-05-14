# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:30:37 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("Largecap_Balancesheet.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values

#Encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("City", OneHotEncoder(), [4])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting
y_pred = regressor.predict(X_test)
print(regressor.score(X_train, y_train))


#Graph
import matplotlib.pyplot as plt

x_serial = list(range(1, len(y_pred) + 1))
plt.scatter(x_serial, y_pred, color = 'blue')
plt.scatter(x_serial, y_test, color = 'red')
plt.title('Comparison between Actual and Predicted value')
plt.xlabel('Serial Number')
plt.ylabel('Operating Profit')
plt.show()














