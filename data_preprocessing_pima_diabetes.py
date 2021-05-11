# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:28:53 2021

@author: RISHBANS
"""

import pandas as pd
df = pd.read_csv("pima-data.csv")

#1. Null Values
df.isnull().values.any()

#2. Corr
corr = df.corr()
del df['skin']

#3. Data Moulding
diabetes_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

#4.Split Train - Test
from sklearn.model_selection import train_test_split

features_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness',
                      'insulin','bmi', 'diab_pred', 'age']
pred_class_name = ['diabetes']

X = df[features_col_names].values
y = df[pred_class_name].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
y_train = y_train.ravel()

#5. Impute missing values
from sklearn.impute import SimpleImputer
fill_0 = SimpleImputer(missing_values = 0, strategy= 'mean')
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

#6. Standardisation
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 15)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=["no", "yes"]))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

classifier.score(X_test,y_test)





















