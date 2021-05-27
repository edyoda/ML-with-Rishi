# -*- coding: utf-8 -*-
"""
Created on Thu May 27 19:30:01 2021

@author: RISHBANS
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv("Job_Exp.csv")
X = dataset.iloc[:, [0]].values
y = dataset.iloc[:, 1].values

dt_r = DecisionTreeRegressor(random_state=0)
dt_r.fit(X, y)

y_pred = dt_r.predict([[30]])
print(y_pred)

X_grid = np.arange(min(X), max(X),0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'green')
plt.plot(X_grid, dt_r.predict(X_grid), color='red')
plt.xlabel("Years in Exp")
plt.ylabel("Getting Job %")