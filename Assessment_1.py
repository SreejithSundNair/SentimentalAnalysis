# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:48:32 2020

@author: SDS7RT
"""

import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import seaborn as sns
from sklearn.model_selection import train_test_split

# Data importing 
data_df = []
for line in open('Beauty_5_50000.json', 'r'):
    data_df.append(json.loads(line))
    
data_df = pd.DataFrame(data_df)

""" Data pre-processing

1. Deleting all the row which has no review text
2. Rows with same reviewID, asin, unixReviewTime were removed.
3. reseting the index value
4. creating labels from 1 to 5 represents the overall score

"""
#data_df = data_df[~pd.isnull(data_df['reviewText'])]
#data_df.drop_duplicates(subset = ['reviewerID', 'asin', 'unixReviewTime'], inplace = True)
#data_df.reset_index(inplace = True)
#data_df['Overall_range'] = pd.cut(x=data_df['overall'], bins=[0, 1, 2, 3, 4, 5],
                                         #labels=['1', '2', '3', '4', '5'], include_lowest=True)
#label_2 = data_df['overall'].factorize()[0]
# return the wordnet object value corresponding to the POS tag
labels = data_df['overall']

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# clean text data
data_df["cleaned_Review"] = data_df["reviewText"].apply(lambda x: clean_text(x))

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=15, norm='l2', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(data_df.cleaned_Review).toarray()
#labels = data_df.Overall_range

from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 40)

model = RandomForestClassifier(n_estimators = 100, random_state = 40)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import average_precision_score, precision_recall_curve
average_precision = average_precision_score(y_test, y_pred)
