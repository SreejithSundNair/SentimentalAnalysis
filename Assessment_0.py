# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:06:26 2020

@author: SDS7RT

"""
import json
import pandas as pd


# Data importing 
data_df = []
for line in open('Beauty_5_50000.json', 'r'):
    data_df.append(json.loads(line))
    
data_df = pd.DataFrame(data_df)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(data_df['reviewText'], data_df['overall'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train).toarray()
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts).toarray()

X_test_counts = count_vect.fit_transform(X_test).toarray()
tfidf_transformer = TfidfTransformer()
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts).toarray()

clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)