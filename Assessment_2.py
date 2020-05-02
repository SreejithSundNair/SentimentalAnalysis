# Importing libraries
import json
from xgboost import XGBClassifier
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sn

# Data importing 
data_df = []
for line in open('Beauty_5_50000.json', 'r'):
    data_df.append(json.loads(line))
   
data_df = pd.DataFrame(data_df)

""" Data pre-processing
1. Deleting all the row which has no review text
2. Rows with same reviewID, asin, unixReviewTime were removed.
3. reseting the index value
"""
data_df = data_df[~pd.isnull(data_df['reviewText'])]
data_df.drop_duplicates(subset = ['reviewerID', 'asin', 'unixReviewTime'], inplace = True)
data_df.reset_index(inplace = True)

# function for returning Part Of Speech
# only considering:- J: Adjective, V: Verb, N: Noun, R: Adverb
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
def preprocess(txt):
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
    return [t for t in txt.split()] 


# Classification and train#
# classes are imbalanced in dataset:- (1: 0.05342, 2:0.05686, 3:0.11362, 4:0.19926, 5:0.57684)
# are the normalized count (weight) of each class
pipeline = Pipeline([('Tf-Idf', TfidfVectorizer(ngram_range = (1,2), analyzer = preprocess)), 
                     ('classifier', XGBClassifier(learning_rate = 0.05, objective = 'multi:softmax',num_class = 5, n_estimators=300))]) 
X = data_df.reviewText
y = data_df.overall
train_df, test_df, lab_train, lab_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
pipeline.fit(train_df, lab_train)
# test and evaluation
Pred = pipeline.predict(test_df)
print(metrics.classification_report(lab_test, Pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(lab_test, Pred)
print(cm)
sn.heatmap(cm, annot = None)

# Updating test review result in a csv file
review_test = pd.DataFrame(data = {'review_test':test_df, 'prediction': Pred})
review_test.to_csv('test_review.csv')

""" Most correlated words capturing"""
# Function to show most correlated words
def show_cloud(words, title = None):
    wordcloud = WordCloud(background_color = 'white', max_words = 50, 
                          max_font_size = 40, scale = 3, random_state = 40).generate(str(words))
    fig = plt.figure(1, figsize= (20, 20))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize = 20)
    fig.subplots_adjust(top = 2.3)
    plt.imshow(wordcloud)
    plt.show()
# To get features
features = pipeline['Tf-Idf'].fit_transform(data_df.reviewText)
# Importing library
from sklearn.feature_selection import chi2
import numpy as np
# Words which has overall <=2
Bad_chi2 = chi2(features, (data_df.overall <= 2))
indices = np.argsort(Bad_chi2[0])
Bad_words = np.array(pipeline['Tf-Idf'].get_feature_names())[indices]
show_cloud(Bad_words[-100:])
#word which has overall >=4
Good_chi2 = chi2(features, (data_df.overall >= 4))
indices_1 = np.argsort(Good_chi2[0])
Good_words = np.array(pipeline['Tf-Idf'].get_feature_names())[indices_1]
show_cloud(Good_words[-100:])