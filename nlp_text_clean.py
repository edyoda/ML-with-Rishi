# -*- coding: utf-8 -*-
"""
Created on Mon May 31 19:55:49 2021

@author: RISHBANS
"""

import pandas as pd

dataset = pd.read_csv("Tweets.csv")

dataset.isnull().values.any()
dataset.isnull().sum(axis=0)
dataset.fillna(0)


import matplotlib.pyplot as plt
#Data analysis
#1. No. of tweets for each airline by percentage
dataset.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')

#2. Sentiment distribution for each indivi airline
dataset_sentiment = dataset.groupby(['airline','airline_sentiment']).airline_sentiment.count().unstack()
dataset_sentiment.plot(kind='bar')


#3. Sentiment distribution
dataset.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%')

feature = dataset.iloc[:, 10].values 

#Text Cleanup using Regular Exp.
import re
process_tweet = []

for tweet in range(0, len(feature)):
    #filter our special characters
    clean_tweet = re.sub(r'\W', ' ', str(feature[tweet]))    
    
    #filter out single characters
    clean_tweet = re.sub('\s+[a-zA-Z]\s+', ' ', clean_tweet)
    #remove numbers
    clean_tweet = re.sub('\s+[0-9]+\s+', ' ', clean_tweet )
    #remove multiple white spaces
    clean_tweet = re.sub('\s+', ' ', clean_tweet)
    #converting to lower number
    clean_tweet = clean_tweet.lower()

    
    
    process_tweet.append(clean_tweet)



##Appendix Code

tweet = 'ðŸ˜Ž RT @VirginAmerica: Youâ€™ve met your match. Got status on another airline? Upgrade (+restr): http://t.co/RHKaMx9VF5. http://t.co/PYalebgkJt'
tweet = re.sub(r'\W', ' ', str(tweet))
print(tweet)
tweet = re.sub('\s+[a-zA-Z]\s+', ' ', tweet)
tweet = re.sub('\s+[0-9]+\s+', ' ', tweet )
print(tweet)
tweet = re.sub('\s+', ' ', tweet)
print(tweet)
tweet = tweet.lower()
print(tweet)






























