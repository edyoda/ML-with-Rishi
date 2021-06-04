# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 20:37:58 2021

@author: RISHBANS
"""

import pandas as pd
dataset = pd.read_csv("Sentiment.csv")

dataset = dataset.drop(dataset[dataset.sentiment == "Neutral"].index)
sent_map = {"Positive": 1, "Negative":0}
dataset["sentiment"] = dataset["sentiment"].map(sent_map)

X = dataset["text"]
y = dataset["sentiment"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

from sklearn.feature_extraction.text import TfidfVectorizer
sent_tfidf = TfidfVectorizer(max_df = 0.8, min_df = 0.001, stop_words='english')

X_train = sent_tfidf.fit_transform(X_train).toarray()
X_test = sent_tfidf.transform(X_test).toarray()

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)
print(accuracy_score(y_test, gnb_pred))

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
mnb_pred = mnb.predict(X_test)

print(accuracy_score(y_test, mnb_pred))









