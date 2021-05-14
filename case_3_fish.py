# -*- coding: utf-8 -*-
"""
Created on Fri May 14 20:28:24 2021

@author: RISHBANS
"""
import pandas as pd

dataset = pd.read_csv("Fish.csv")

X = dataset.iloc[:, 2:7].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.25, random_state = 12)

#Applying Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

print(linear_model.score(X_test, y_test))