# -*- coding: utf-8 -*-
"""
Created on Fri May 14 19:30:18 2021

@author: RISHBANS
"""

from sklearn.datasets import california_housing
data_h = california_housing.fetch_california_housing()
print(data_h.data)
print(data_h.target)

import pandas as pd
house_data = pd.DataFrame(data_h.data, columns = data_h.feature_names)
house_data['Price'] = data_h.target

#Store in X and y
X = house_data.iloc[:, 0:8].values
y = house_data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 1)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

print(linear_model.score(X_test, y_test))