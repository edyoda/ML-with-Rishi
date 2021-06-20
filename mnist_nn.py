# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 20:16:15 2021

@author: RISHBANS
"""

import pandas as pd
mnist_data = pd.read_csv("mnist-train.csv")


features = mnist_data.columns[1:]

X = mnist_data[features]
y = mnist_data['label']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X/255, y, test_size =0.15, random_state = 0)

import numpy as np
from keras.utils import np_utils

print(np.unique(y_train, return_counts = True))

n_classes = 10

y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)


from keras.models import Sequential
from keras.layers import Dense, Dropout

mnist_nn = Sequential()

#Hidden Layer
mnist_nn.add(Dense(units = 100, kernel_initializer='uniform', activation = 'relu', input_dim=784))

mnist_nn.add(Dropout(0.2))

mnist_nn.add(Dense(units = 10, kernel_initializer='uniform', activation = 'softmax'))

mnist_nn.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])

mnist_nn.fit(X_train, y_train, batch_size = 64, epochs = 20, validation_data = (X_test, y_test))


y_pred = mnist_nn.predict(X_test)
y_pred = ( y_pred > 0.9)



