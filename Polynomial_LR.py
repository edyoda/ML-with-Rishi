# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:50:34 2021

@author: RISHBANS
"""

import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("Company_Performance.csv")

X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, 1].values

#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Polynomial
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

poly_lin_reg = LinearRegression()
poly_lin_reg.fit(X_poly, y)
y_pred = poly_lin_reg.predict(X_poly)

#Simple Linear Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'green')
plt.title('Size of Company')
plt.xlabel('No. of years of Operation')
plt.ylabel('No. of Employees')
plt.show()


#Polynomial Linear Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred, color = 'green')
plt.title('Size of Company(Polynomial Linear Regression)')
plt.xlabel('No. of years of Operation')
plt.ylabel('No. of Employees')
plt.show()





