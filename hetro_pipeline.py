# -*- coding: utf-8 -*-
"""
Created on Mon May 24 19:31:33 2021

@author: RISHBANS
"""

import pandas as pd
hr_data = pd.read_csv("HR_comma_sep.csv")
hr_data.rename(columns={'sales':'dept'}, inplace=True)

X = hr_data.drop(columns=['left'])
y = hr_data.left

print(X.dtypes)
#obj_data = X.select_dtypes(include=['object'])
#dept = onehot, salary=ordinal
#int(5) = minmaxscaler, selectkbest

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

obj1_pipeline = make_pipeline(OrdinalEncoder())
obj2_pipeline = make_pipeline(OneHotEncoder())
int_pipeline = make_pipeline(MinMaxScaler(), SelectKBest(k=3, score_func= f_classif))

preprocessor = make_column_transformer(
               (obj1_pipeline, ['salary']),
               (obj2_pipeline, ['dept']),
               (int_pipeline, ['number_project', 'average_montly_hours', 'time_spend_company']),
               remainder = 'passthrough'    
    )

m_pipeline = make_pipeline(preprocessor, RandomForestClassifier())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

m_pipeline.fit(X_train, y_train)

y_pred = m_pipeline.predict(X_test)
print(m_pipeline.score(X_test, y_test))

print(m_pipeline.steps)

from sklearn.model_selection import GridSearchCV
params = {'columntransformer__pipeline-3__selectkbest__k': [1,2,3]}

gs = GridSearchCV(m_pipeline, param_grid=params, cv = 4)
gs.fit(X_train, y_train)
print(gs.best_params_)
print(gs.best_score_)
print(gs.score(X_test, y_test))






















