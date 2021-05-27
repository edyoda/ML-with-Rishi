# -*- coding: utf-8 -*-
"""
Created on Thu May 27 20:11:40 2021

@author: RISHBANS
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
#import data
iris = load_iris()

#Split data into X and y
X = iris.data
y = iris.target


#Split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#Build the model
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)

#check accuracy score
y_pred = dt.predict(X_test)
from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, y_pred)
print(test_acc)



