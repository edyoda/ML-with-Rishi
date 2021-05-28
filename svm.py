# -*- coding: utf-8 -*-
"""
Created on Fri May 28 20:00:17 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("BankNote_Authentication.csv")

X = dataset.iloc[:, [0,1]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#SVM
from sklearn.svm import SVC
svm = SVC(kernel='sigmoid', random_state=0)
svm.fit(X_train, y_train)

pred_test = svm.predict(X_test)

from sklearn.metrics import accuracy_score

test_accu = accuracy_score(y_test, pred_test)
print(test_accu)


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
#Define Variables
clf = svm
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
plt.title('Support Vector Machine-sigmoid')
plt.xlabel('Independent Variable 1')
plt.ylabel('Independent Variable 2')
plt.legend()













