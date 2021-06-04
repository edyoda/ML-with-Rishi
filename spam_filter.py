# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 20:15:29 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("emails.csv")

X = dataset["text"]
y = dataset["spam"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.feature_extraction.text import CountVectorizer
spam_fil = CountVectorizer(stop_words = 'english')
X_train = spam_fil.fit_transform(X_train).toarray()
X_test = spam_fil.transform(X_test).toarray()

print(spam_fil.get_feature_names())

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
kneigh = KNeighborsClassifier(n_neighbors=200)
kneigh.fit(X_train, y_train)
kneigh_test = kneigh.predict(X_test)

print(accuracy_score(y_test, kneigh_test))


from sklearn.naive_bayes import GaussianNB
g_nb = GaussianNB()
g_nb.fit(X_train, y_train)
g_nb_pred = g_nb.predict(X_test)

print(accuracy_score(y_test, g_nb_pred))



from sklearn.naive_bayes import MultinomialNB
m_nb = MultinomialNB()
m_nb.fit(X_train, y_train)
m_nb_pred = m_nb.predict(X_test)

print(accuracy_score(y_test, m_nb_pred))



from sklearn.naive_bayes import BernoulliNB
b_nb = BernoulliNB()
b_nb.fit(X_train, y_train)
b_nb_pred = b_nb.predict(X_test)

print(accuracy_score(y_test, b_nb_pred))















