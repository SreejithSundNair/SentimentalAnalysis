import json
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
#from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
data_df = data_df[~pd.isnull(data_df['reviewText'])]
data_df.drop_duplicates(subset = ['reviewerID', 'asin', 'unixReviewTime'], inplace = True)
data_df.reset_index(inplace = True)

data_df['Overall_range'] = pd.cut(x=data_df['overall'], bins=[0, 1, 2, 3, 4, 5],
                                         labels=['1', '2', '3', '4', '5'], include_lowest=True)

# function for returning part of speech
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
# Function to preprocess the text data 
def preprocess(txt): #reviewTxt):
    # lowercase
    txt = txt.lower()
    # tokenize and remove puncutation
    txt = [word.strip(string.punctuation) for word in txt.split(" ")]
    # remove numbers in and as word
    txt = [word for word in txt if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    txt = [x for x in txt if x not in stop]
    # remove empty tokens
    txt = [t for t in txt if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(txt)
    # lemmatize text
    txt = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    txt = [t for t in txt if len(t) > 1]
    # join all
    txt = " ".join(txt)
    return txt


data_df['cleaned_review'] = data_df['reviewText'].apply(lambda x: preprocess(x))