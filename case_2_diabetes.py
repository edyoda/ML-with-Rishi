# -*- coding: utf-8 -*-
"""
Created on Fri May 14 19:57:54 2021

@author: RISHBANS
"""

from sklearn.datasets import load_diabetes
data = load_diabetes()

import pandas as pd
diab_data = pd.DataFrame(data.data, columns = data.feature_names)
diab_data['Price'] = data.target

#Store in X and y
X = diab_data.iloc[:, 0:10].values
y = diab_data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.25, random_state = 12)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

print(linear_model.score(X_test, y_test))