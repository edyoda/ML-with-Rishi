# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:35:09 2021

@author: RISHBANS
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

dataset = pd.read_csv("titanic3.csv")
print(dataset.isnull().sum(axis = 0))

dataset.drop(['cabin', 'boat', 'body', 'home.dest', 'name', 'ticket'], axis = 1, inplace=True)
print(dataset.dtypes)

X = dataset.drop(columns=['survived'])
y = dataset.survived

#sex, embarked =>  simpleimputer (most_frequent), onehot
#age, fare => simpleimputer (mean), scalar

pipeline_1 = make_pipeline(SimpleImputer(missing_values = np.nan, strategy = 'most_frequent'),
                           OneHotEncoder())
pipeline_2 = make_pipeline(SimpleImputer(missing_values = np.nan, strategy = 'mean'),
                           StandardScaler())

preprocessor = make_column_transformer(
               (pipeline_1, ['sex', 'embarked']),
               (pipeline_2, ['age', 'fare']),
               remainder = 'passthrough'
    )

m_pipeline = make_pipeline(preprocessor, SelectKBest(k=3, score_func=f_classif),RandomForestClassifier())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

m_pipeline.fit(X_train, y_train)

print(m_pipeline.score(X_test, y_test))
print(m_pipeline.steps)

#applying grid search
from sklearn.model_selection import GridSearchCV
params = {'selectkbest__k':[1,2,3,4, 8]}
gs = GridSearchCV(m_pipeline, param_grid=params, n_jobs=4, cv=5)
gs.fit(X_train, y_train)
print(gs.best_params_)
print(gs.n_features_in_)
print(gs.best_score_)
print(gs.score(X_test, y_test))








