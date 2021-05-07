# -*- coding: utf-8 -*-
"""
Created on Fri May  7 19:29:22 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

from sklearn.cluster import KMeans
k_means = KMeans(n_clusters = 5, init = 'k-means++', random_state = 1)
print(k_means.fit_predict(X))


wcss = []
for i in range(1, 15):
    k_means = KMeans(n_clusters = i, init = 'k-means++', random_state = 1)
    k_means.fit_predict(X)
    wcss.append(k_means.inertia_)
    print("i=", i, "wcss=", k_means.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1,15), wcss)
plt.title('Elbow Method')
plt.xlabel('Number Of Clusters')
plt.ylabel('wcss score')
plt.show()