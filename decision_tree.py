# -*- coding: utf-8 -*-
"""
Created on Wed May 26 20:44:13 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("Apply_Job.csv")

X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
dt_c = DecisionTreeClassifier()
dt_c.fit(X_train, y_train)

pred_test = dt_c.predict(X_test)

from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(y_test, pred_test)

print(test_accuracy)

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
#Define Variables
clf = dt_c
h = 0.1
X_plot, z_plot = X_train, y_train 

#Standard Template to draw graph
x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh
Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z,
             alpha = 0.7, cmap = ListedColormap(('blue', 'red')))


for i, j in enumerate(np.unique(z_plot)):
    plt.scatter(X_plot[z_plot == j, 0], X_plot[z_plot == j, 1],
                c = ['blue', 'red'][i], cmap = ListedColormap(('blue', 'red')), label = j)
   #X[:, 0], X[:, 1] 
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('Decision Tree')
plt.xlabel('Exp in Year')
plt.ylabel('Salary in Lakh')
plt.legend()