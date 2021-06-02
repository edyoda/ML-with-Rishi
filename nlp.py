# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 19:33:24 2021

@author: RISHBANS
"""
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem.lancaster import LancasterStemmer
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import wordnet
from nltk.wsd import lesk

#1. Tokenization
text = "I believe this would help the reader understand how tokenization \
        works. as well as realize its importance."
        
sents = sent_tokenize(text)
print(sents)

words = [word_tokenize(sent) for sent in sents]
print(words)

word_com = word_tokenize(text)
print(word_com)

#2. Stop word removal
text = "I believe this would help the reader understand how tokenization \
        works. as well as realize its importance (text) ."
        
custom_list = set(stopwords.words('english')  + list(punctuation))
#list comprehension
word_list = [word for word in word_tokenize(text) if word not in custom_list]
print(word_list)

#3.Stemming
l_s = LancasterStemmer()
new_text = "It is important to by very pythonly while you are pythoning\
             with python. All pythoners have pythoned poorly at least once."

stem_lan = [ l_s.stem(word) for word in word_tokenize(new_text)]
print(stem_lan)

#4. N-Grams
word_list = ['it', 'is', 'import', 'to', 'by', 'very', 'python', 'whil', 'you',\
             'ar', 'python', 'with', 'python', '.', 'al', 'python', 'hav', \
                 'python', 'poor', 'at', 'least', 'ont', '.']
finde = BigramCollocationFinder.from_words(word_list)
print(finde.ngram_fd.items())   

#5. WSD
for ss in wordnet.synsets('watch'):
    print(ss, ss.definition())
    
context = lesk(word_tokenize("i cant bear now, its too much for me"), "bear")
print(context, context.definition())

#6.  count vectorizer
import pandas as pd
corpus = [
     'This is the first document from heaven',
     'but the second document is from mars',
     'And this is the third one from nowhere',
     'Is this the first document from nowhere?',
]    
    
df = pd.DataFrame({'text': corpus})

from sklearn.feature_extraction.text import CountVectorizer
count_v = CountVectorizer()
X = count_v.fit_transform(df.text).toarray() 
print(X)   
    
count_v = CountVectorizer(stop_words = ['this','is'])
X = count_v.fit_transform(df.text).toarray() 
print(X)  
print(count_v.vocabulary_)     
    
# 7. Hashing
from sklearn.feature_extraction.text import HashingVectorizer

df = pd.DataFrame({'text': corpus})
hash_v = HashingVectorizer(n_features = 1000)
Y = hash_v.fit_transform(df.text).toarray()
print(Y)

#8. TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
corpus = [[
     'This is the first document from heaven',
     'but the second document is from mars',
     'And this is the third one from nowhere',
     'Is this the first document from nowhere?',
]]
df = pd.DataFrame({'text': corpus})

vector = TfidfTransformer()
vector.fit(corpus)
print(vector.vocabulary_)
    



















