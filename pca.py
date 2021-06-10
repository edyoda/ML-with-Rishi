# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 20:08:09 2021

@author: RISHBANS
"""

#import libraries
import pandas as pd

#import data
df = pd.read_csv("pima-data.csv")

#check the correlation
df.corr()
del df['skin']

#Data Molding
diab_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diab_map)

#Store values in X and y
X = df.iloc[:, 0:8]
y = df.iloc[:, 8]


#Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

#Simple Imputer
from sklearn.impute import SimpleImputer
fill_0 = SimpleImputer(missing_values=0, strategy='mean')

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.transform(X_test)


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


#applying pca
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explain_var = pca.explained_variance_ratio_
print(explain_var)

#Plot to understand variance
#import matplotlib.pyplot as plt
#import numpy as np
#plt.plot(np.cumsum(explain_var))


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
#Define Variables
clf = gnb
h = 0.01
X_plot, z_plot = X_test, y_test 

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
plt.title('Naive Bayes with PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

plt.show()

















