# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:32:01 2021

@author: RISHBANS
"""

import pandas as pd
exam_data = pd.read_csv('exams.csv')

from sklearn import preprocessing
#scaling
exam_data[['math score']] = preprocessing.scale(exam_data[['math score']])
exam_data[['reading score']] = preprocessing.scale(exam_data[['reading score']])
exam_data[['writing score']] = preprocessing.scale(exam_data[['writing score']])

#Label Encoding
le = preprocessing.LabelEncoder()
exam_data['gender'] = le.fit_transform(exam_data['gender'].astype(str))
exam_data.head()

#One Hot Encoding
exam_data = pd.get_dummies(exam_data, ['race/ethnicity', 'parental level of education',
                                       'lunch', 'test preparation course'])

del exam_data['race/ethnicity_group E']


X = exam_data.iloc[:, 1: 18 ].values
y = exam_data.iloc[:, 0].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

from sklearn.neighbors import KNeighborsClassifier
classfier = KNeighborsClassifier(n_neighbors = 9)
classfier.fit(X_train, y_train)

y_pred = classfier.predict(X_test)

print(classfier.score(X_test, y_test))







