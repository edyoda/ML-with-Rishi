# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:33:11 2021

@author: RISHBANS
"""

import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()

# Store in DataFrame
dataset = pd.DataFrame(boston.data, columns = boston.feature_names)
dataset['price'] = boston.target
corr = dataset.corr()

del dataset['TAX']
# Split into X and y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Split into Train and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Standardization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(units = 12, kernel_initializer='uniform', activation = 'relu', input_dim = 12))

model.add(Dense(units = 1, kernel_initializer='uniform'))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])

model.fit(X_train, y_train, batch_size = 4, epochs = 150)
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

















