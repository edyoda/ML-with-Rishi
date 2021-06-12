# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 21:00:25 2021

@author: RISHBANS
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)


from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=10)),
    ('svc', SVC(kernel='rbf')),
    ('knc', KNeighborsClassifier()),
    ('abc', AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=20)),  
    ('lr', LogisticRegression())
    ]

vc = VotingClassifier(estimators=estimators, voting= 'hard')
vc.fit(X_train, y_train)

for est,name in zip(vc.estimators_, vc.estimators):
    print(name[0], est.score(X_test, y_test))
print(vc.score(X_test, y_test))  
    
  
    
  
    
  
    
  