# -*- coding: utf-8 -*-
"""
Created on Wed May  5 19:29:57 2021

@author: RISHBANS
"""
#Implement KNN from scratch

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['type'] = iris.target

class MyKNN:
    def __init__(self,k):
        self.k = k
        
    #Convert and assign feature and target data to variable
    def my_fit(self, feature_data, target_data):
        self.feature_data = np.array(feature_data)
        self.target_data = np.array(target_data)
        
    #Calculate Euclidean Distance
    def calculate_distance_vector(self, one_data):
        distances = np.sqrt(np.sum(np.square(self.feature_data - one_data), axis = 1))
        return distances
    
    #Sort the distance and take min value (k)
    def find_k_neighbours(self, one_data_feature):
        res = self.calculate_distance_vector(one_data_feature)
        return res.argsort()[:self.k]
    
    #find the index of the closest neighbours
    def find_k_neighbours_index(self, one_data_feature):
        index_of_neighbour = self.find_k_neighbours(one_data_feature)
        return self.target_data[index_of_neighbour]
    
    #my prediction - based on majority
    def my_predict(self, one_data_feature):
        classes = self.find_k_neighbours_index(one_data_feature)
        print(classes)
        return np.bincount(classes).argmax()
    
model =  MyKNN(k=5)
feature_data =  df.drop(columns=['type'], axis=1)
target_data = df.type

model.my_fit(feature_data, target_data)    

one_data = [1,2,3,4]
model.find_k_neighbours_index(one_data)  
print(model.my_predict(one_data))  
    
    
    
    
#Example to explain calculate_distance_vector - axis = 1    
a = np.array([[2,1,3,4], [4,3,2,1]])
one_data = [1,2,3,4]    
print(a - one_data)
print(np.square(a - one_data))
print(np.sum(np.square(a - one_data),axis = 1))
print(np.sqrt(np.sum(np.square(a - one_data),axis = 1)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

