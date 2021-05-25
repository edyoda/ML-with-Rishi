# -*- coding: utf-8 -*-
"""
Created on Tue May 25 19:56:17 2021

@author: RISHBANS
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_features = 2, n_samples = 1000, cluster_std=0.8, centers=4)
plt.scatter(X[:,0], X[:, 1], s=10, c=y)
plt.xlabel('X[0]')
plt.ylabel('X[1]')
plt.colorbar()

import pandas as pd
df = pd.DataFrame({'X1':X[:,0], 'X2':X[:,1]})
df['target'] = y

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')

dt.fit(df[['X1','X2']], df.target)

import numpy as np
plot_step = 0.1
x_min, x_max = X[:, 0].min() -1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() -1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

predict = dt.predict(np.c_[xx.ravel(), yy.ravel()])


plt.scatter(X[:,0], X[:, 1], s=10, c=y)
plt.xlabel('X[0]')
plt.ylabel('X[1]')
plt.colorbar()
plt.scatter(xx.ravel(), yy.ravel(), c=predict, s=0.5, alpha = 0.1)

########################################################
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()

knc.fit(df[['X1','X2']], df.target)

plot_step = 0.1
x_min, x_max = X[:, 0].min() -1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() -1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

predict = knc.predict(np.c_[xx.ravel(), yy.ravel()])


plt.scatter(X[:,0], X[:, 1], s=10, c=y)
plt.xlabel('X[0]')
plt.ylabel('X[1]')
plt.colorbar()
plt.scatter(xx.ravel(), yy.ravel(), c=predict, s=0.5, alpha = 0.1)

#################################################################
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(df[['X1','X2']], df.target)

plot_step = 0.1
x_min, x_max = X[:, 0].min() -1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() -1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

predict = lr.predict(np.c_[xx.ravel(), yy.ravel()])


plt.scatter(X[:,0], X[:, 1], s=10, c=y)
plt.xlabel('X[0]')
plt.ylabel('X[1]')
plt.colorbar()
plt.scatter(xx.ravel(), yy.ravel(), c=predict, s=0.5, alpha = 0.1)


















