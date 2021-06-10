# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 20:35:17 2021

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


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
explain_var = lda.explained_variance_ratio_
print(explain_var)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))






