# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 19:20:42 2021

@author: RISHBANS
"""
import pandas as pd
dataset = pd.read_csv("horror-train.csv")

X = dataset.text
y = dataset.author

#Text Preprocessing
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\s[A-Za-z]+\s')

dataset['text_tokenized'] = dataset.text.map(lambda t: tokenizer.tokenize(t))

#Stemming
from nltk.stem import PorterStemmer
l_s = PorterStemmer()
dataset['text_stemmed'] = dataset['text_tokenized'].map(lambda l: [l_s.stem(word) for word in l])

#join the words
dataset['text_sent'] = dataset['text_stemmed'].map(lambda l: ''.join(l))
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(stop_words='english')
X = dataset.text_sent
y = dataset.author

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
spam_fil = CountVectorizer(stop_words='english')
X_train = spam_fil.fit_transform(X_train).toarray()
X_test = spam_fil.transform(X_test).toarray()

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

print(mnb.score(X_test, y_test))


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

print(bnb.score(X_test, y_test))

