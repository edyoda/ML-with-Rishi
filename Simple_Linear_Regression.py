# -*- coding: utf-8 -*-
"""
Created on Wed May 12 19:12:26 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("Company_profit.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print(linear_model.score(X_train, y_train))

y_pred = linear_model.predict(X_test)

print(linear_model.score(X_test, y_test))

#Visualization
import matplotlib.pyplot as plt

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, linear_model.predict(X_train), color = 'green')
plt.title('Training Set Graph')
plt.xlabel('Startup Yrs Operation')
plt.ylabel('Profit')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, linear_model.predict(X_train), color = 'green')
plt.title('Test Set Graph')
plt.xlabel('Startup Yrs Operation')
plt.ylabel('Profit')
plt.show()


x_serial = list(range(1, len(y_pred) + 1))
plt.scatter(x_serial, y_pred, color = 'blue')
plt.scatter(x_serial, y_test, color = 'red')
plt.title('Comparison between Actual and Predicted value')
plt.xlabel('Serial Number')
plt.ylabel('Profit')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.scatter(X_test, y_pred, color = 'blue')
plt.title('Comparison Graph - Linear')
plt.xlabel('Startup Yrs Operation')
plt.ylabel('Profit')
plt.show()











