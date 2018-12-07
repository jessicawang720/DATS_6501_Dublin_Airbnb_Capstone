
# Load required libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans  # MiniBatchKMeans really helps to fasten processing time
from nltk import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import math as math
from scipy.misc import imread
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords 
stop_words = set(stopwords.words("english"))
import re
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud,ImageColorGenerator


# ### Function and class definitions


class LemmaTokenizer(object):
    """Custom tokenizer class that stems tokens"""
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self,doc):
        return [self.wnl.lemmatize(t) for t in wordpunct_tokenize(doc) if len(t.strip()) > 1]
    
def show_topn(classifier,vectorizer,categories,n):
    """Returns the top n features that characterize eachc category"""
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        topn = np.argsort(classifier.coef_[i])[-n:]
        print('{}: {}'.format(category,", ".join(feature_names[topn])))
        
def save_topn(classifier,vectorizer,categories,n,outdict):
    """Returns the top n features that characterize eachc category, and save result in outdict"""
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        topn = np.argsort(classifier.coef_[i])[-n:]
        outdict[i] = feature_names[topn]

# read in a few columns from the data and show the top of the resulting dataframe
df = pd.read_csv('Dublin_Airbnb_listings.csv', usecols = ['id', 'name', 'space', 'description','neighbourhood','neighborhood_overview', 'neighbourhood_cleansed'])
#df = df.dropna(axis=0, subset=['neighbourhood'])
df = df.fillna('a')
df.head()


# # df.head()


# Check the full text in each of the column
for i in range(len(df.columns)):
    print(df.columns[i],": ")
    print(df.iloc[0,i])
    print('=======================')

# let's combine the name, space, description, and neighborhood_overview into a new column
df['all_description'] = df.apply(lambda x: '{} {} {} {}'.format(x['name'], x['space'], x['description'], x['neighborhood_overview']), axis=1)
print(df.loc[0,'all_description'])

# Text cleaning: Remove punctuations, emoji and numbers; Lemmatizer; Convert each word to its lower case;Tokenization;Stopwords removal;

Descriptions_text = [] 
for i in range(0, len(df)):    
    # column : "Descriptions Text", row ith 
    descriptions = re.sub('[^a-zA-Z]', ' ', str(df['all_description'][i]))      
    # convert all cases to lower cases 
    descriptions = descriptions.lower()      
    # split to array(default delimiter is " ") 
    descriptions = descriptions.split()      
    # creating wordnet_lemmatizer object to take main stem of each word 
    wordnet_lemmatizer = WordNetLemmatizer()      
    # loop for lemmatizer each word in string array at ith row     
    descriptions = [wordnet_lemmatizer.lemmatize(word) for word in descriptions if not word in stop_words]                  
    # rejoin all string array elements to create back into a string 
    descriptions = ' '.join(descriptions)      
    # append each string to create array of clean text  
    Descriptions_text.append(descriptions)

# combine corpus into original train dataframe
df['Descriptions_text'] = pd.DataFrame(Descriptions_text)


print(df.loc[4,'Descriptions_text'])

# Transform combined_description into tfidf format

tf_idf = TfidfVectorizer(ngram_range=(1,2),stop_words='english',tokenizer=LemmaTokenizer())
tf_idf.fit(df['Descriptions_text'])
DescTfidf = tf_idf.transform(df['Descriptions_text'])


# **How many listings are there for each neighborhood?**

# I added a chart to replace tabulation in the original notebook

dublin_neighbors = df.groupby(by='neighbourhood').count()[['id']].sort_values(by='id', ascending=False)
#remove 'a' row
dublin_neighbors = dublin_neighbors.drop(['a'])
# print(neighborRank)
plt.figure(figsize=(10,10))
g = sns.barplot(y=dublin_neighbors.index,x=dublin_neighbors["id"])
# The line below adds the value label in each bar
[g.text(p[1]+1,p[0],p[1], color='red') for p in zip(g.get_yticks(), dublin_neighbors["id"])]
plt.title('Dublin Airbnb listings in Each Neighborhood')


# # **K-Means Clustering**
# ###  Four segments
# 

# Create K-Means using MiniBatchKMeans. The MiniBatch version works much faster than regular KMeans
k_means4 = MiniBatchKMeans(n_clusters=4)
Descriptionskmeans4 = k_means4.fit_predict(DescTfidf.todense())

# Combine description, cluster, and neighborhood into one dataframe. 
FullDescKmeans4 = pd.concat([pd.DataFrame(Descriptionskmeans4),df[['Descriptions_text','neighbourhood']]],axis=1)
FullDescKmeans4.columns = ['Cluster','Description','Neighbourhood']  
print(FullDescKmeans4.head())


# ### How many listings in each cluster?

# Show and plot the number of listings in each cluster
ClusterNumbers = FullDescKmeans4['Cluster'].value_counts().sort_index()
ClusterNumbers = pd.DataFrame(ClusterNumbers)
ClusterNumbers.columns=['NumListings']
g = sns.barplot(x=FullDescKmeans4['Cluster'].value_counts().index,y=FullDescKmeans4['Cluster'].value_counts())
[g.text(p[0]-0.15,p[1]+5,p[1], color='black') for p in zip(g.get_xticks(), ClusterNumbers["NumListings"])]
plt.title('Number of Listings in Each Description-based Clusters')

# Create crosstab between Cluster and Neighbourhood 
ctab = pd.crosstab(index=FullDescKmeans4['Neighbourhood'],columns=FullDescKmeans4['Cluster'])
#remove 'a' row
ctab = ctab.drop(['a'])
plt.figure(figsize=(12,20))
sns.heatmap(ctab,annot=True,cmap='Blues', fmt='g')
plt.title("Crosstab of Cluster and Dublin Neighbourhood")

