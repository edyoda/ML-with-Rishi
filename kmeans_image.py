# -*- coding: utf-8 -*-
"""
Created on Fri May  7 20:55:33 2021

@author: RISHBANS
"""

from skimage.io import imread, imshow
imshow('car.jpg')

img = imread('car.jpg')
print(img.shape)
print(img)
img = img/255
print(img)
print(img.shape)

from sklearn.cluster import KMeans


kmeans = KMeans(n_clusters = 6, init = 'k-means++')
img_tf = img.reshape(1280*1920, 3)
print(img_tf.shape)
kmeans.fit(img_tf)
kmeans.cluster_centers_


img_com = kmeans.cluster_centers_[[kmeans.labels_]] 
imshow(img_com.reshape(1280, 1920, 3))

