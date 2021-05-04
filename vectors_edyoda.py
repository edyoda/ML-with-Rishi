# -*- coding: utf-8 -*-
"""
Created on Tue May  4 19:53:42 2021

@author: RISHBANS
"""

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances
import numpy as np

X = [[0,1], [1,1]]
y = [[1,2], [2,2]]
print(euclidean_distances(X, y))

print(manhattan_distances(X, y))

print(cosine_distances(X, y))


