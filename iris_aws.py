# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 19:51:35 2021

@author: RISHBANS
"""

from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
df = pd.DataFrame(iris.data)

df.columns = iris.feature_names
df['species'] = iris.target

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

import joblib
joblib.dump(model, "iris-trained-model.pkl")