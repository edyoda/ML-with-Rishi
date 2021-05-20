# -*- coding: utf-8 -*-
"""
Created on Thu May 20 19:23:57 2021

@author: RISHBANS
"""

from sklearn.datasets import load_digits
dataset = load_digits()
dataset.data.shape
X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
trainX_tf = ss.fit_transform(X_train)
testX_tf = ss.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classfier_rf = RandomForestClassifier()
classfier_rf.fit(trainX_tf, y_train)
classfier_rf.predict(testX_tf)


from sklearn.pipeline import make_pipeline
digit_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())

digit_pipeline.fit(X_train, y_train)
y_pred = digit_pipeline.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

digit_pipeline.steps[1][1].feature_importances_


from sklearn.feature_selection import SelectKBest, f_classif
digit_pipeline = make_pipeline(StandardScaler(), SelectKBest(k=10, score_func=f_classif),
                               RandomForestClassifier(n_estimators=100))

print(digit_pipeline)

digit_pipeline.fit(X_train, y_train)
y_pred = digit_pipeline.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


from sklearn.model_selection import GridSearchCV
params = {'selectkbest__k': [10, 20, 30, 40, 50, 60], 
          'randomforestclassifier__n_estimators': [100, 200, 250]}

gs = GridSearchCV(digit_pipeline, param_grid=params, cv = 5)

gs.fit(X_train, y_train)

print(gs.best_params_)
print(gs.best_score_)
y_pred = gs.predict(X_test)
print(y_pred)

print(gs.best_estimator_)






























