# -*- coding: utf-8 -*-
"""
Created on Tue May 25 20:45:59 2021

@author: RISHBANS
"""

import pandas as pd
tennis_data = pd.read_csv("tennis.csv")

from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

o_e = OrdinalEncoder()
X = tennis_data.drop(columns=['play'])
y = tennis_data.play

X = o_e.fit_transform(X)

dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X, y)
print(o_e.categories_)

dt.predict([[1,0,1,0]])

