# # Foreword
# 
#  This notebook takes the same pre-processing step as the original notebook, but took a different direction in analysis. Instead of describing neighrbohoods, this notebook attempts a two-step processes: (1) Create clusters of rooms based on how the owners describe the rooms, (2) Take top 20 words that best describe each cluster. We will take a look at clustering based on 4-clusters and 12-clusters
# 

# # **Pre-processing**

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


# ### Let's get on with exploring

# read in a few columns from the data and show the top of the resulting dataframe
df = pd.read_csv('Dublin_Airbnb_listings.csv', usecols = ['id', 'name', 'space', 'description', 'neighborhood_overview', 'neighbourhood_cleansed'])
df = df.fillna('a')
df.head()

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


# # Green Shamrock Word Cloud
alldata = str(df['Descriptions_text'] )
bg_pic = imread('ireland map.jpg')
wc = WordCloud(mask=bg_pic,background_color='black',scale=20,max_words=2000).generate(alldata)
image_colors = ImageColorGenerator(bg_pic)
wc.recolor(color_func = image_colors)
plt.figure(figsize=(12,25))
plt.imshow(wc)
plt.axis("off")      
plt.show()


# **How many listings are there for each neighborhood?**

# I added a chart to replace tabulation in the original notebook

dublin_neighbors = df.groupby(by='neighbourhood_cleansed').count()[['id']].sort_values(by='id', ascending=False)
# print(neighborRank)
plt.figure(figsize=(10,10))
g = sns.barplot(y=dublin_neighbors.index,x=dublin_neighbors["id"])
[g.text(p[1]+1,p[0],p[1], color='red') for p in zip(g.get_yticks(), dublin_neighbors["id"])]
plt.title('Dublin Airbnb listings in Each Neighborhood')


# # **K-Means Clustering**
# ###  Four segments
# 

# Create K-Means using MiniBatchKMeans. The MiniBatch version works much faster than regular KMeans
k_means4 = MiniBatchKMeans(n_clusters=4)
Descriptionskmeans4 = k_means4.fit_predict(DescTfidf.todense())

# Combine description, cluster, and neighborhood into one dataframe. 
FullDescKmeans4 = pd.concat([pd.DataFrame(Descriptionskmeans4),df[['Descriptions_text','neighbourhood_cleansed']]],axis=1)
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


# Looks like the description clustering is imbalanced, with majority of descriptions fall into cluster 3 and 4

# ### Any close linkage between cluster and neighborhood?

# Create crosstab between Cluster and Neighbourhood 
ctab = pd.crosstab(index=FullDescKmeans4['Neighbourhood'],columns=FullDescKmeans4['Cluster'])
plt.figure(figsize=(10,10))
sns.heatmap(ctab,annot=True,cmap='Blues', fmt='g')
plt.title("Crosstab of Cluster and Dublin Neighbourhood")


# ### Examine the full description of a couple of samples in each cluster

# Let's take a look at the full description from a couple of listings
for i in range(3):
    subset = FullDescKmeans4[FullDescKmeans4['Cluster']==i]
    print('We are at cluster..')
    print(i)
    for j in range(1):
        print(subset.iloc[j,1])
        print('--------------------------------')


# ### Top words that describe each cluster

# Pipeline to identify top 30 words that are "best predictor" of a cluster
pipeline = Pipeline([('tf_idf', TfidfVectorizer(ngram_range=(1,2), stop_words='english', tokenizer=LemmaTokenizer())),
                     ('CLF', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])

modelSegment = pipeline.fit(df['Descriptions_text'],FullDescKmeans4['Cluster'])

#Top 300 words that describe each cluster
Keywords4 = {}
bg_pic1 = imread('green shamrock.jpg')
save_topn(modelSegment.named_steps['CLF'], modelSegment.named_steps['tf_idf'], [str(i) for i in range(4)], 300,outdict=Keywords4)
image_colors = ImageColorGenerator(bg_pic)
fig,axes=plt.subplots(2,2,figsize=(30,30))
wc.recolor(color_func = image_colors)
for i in range(4):
    wordlist = list(Keywords4[i])
    wc = WordCloud(mask=bg_pic1,background_color='black',scale=20,max_words=300).generate(" ".join(wordlist))
    #wc = WordCloud(background_color='black',scale=20,max_words=300).generate(" ".join(wordlist))
    #print(wc)
    axes[math.floor(i/2),i%2].imshow(wc)
    axes[math.floor(i/2),i%2].set_title('cluster: ' + str(i))          
# Create wordcloud based on top-30 words
Keywords4 = {}

save_topn(modelSegment.named_steps['CLF'], modelSegment.named_steps['tf_idf'], [str(i) for i in range(4)], 30,outdict=Keywords4)
fig,axes=plt.subplots(2,2,figsize=(30,12))
for i in range(4):
    wordlist = list(Keywords4[i])
    Word_Cloud = WordCloud(background_color='tan',max_words=30,relative_scaling=0.2).generate(" ".join(wordlist))
    print(Word_Cloud)
    axes[math.floor(i/2),i%2].imshow(Word_Cloud)


# # **Cluster with 12 segments**

# Now we'll try to create a cluster of 12. Hopefully we could get a well spread clustering
kmeans12 = MiniBatchKMeans(n_clusters=12,batch_size=128)
DescKmeans12 = kmeans12.fit_predict(DescTfidf.todense())

FullDescKmeans12 = pd.concat([pd.DataFrame(DescKmeans12),df[['Descriptions_text','neighbourhood_cleansed']]],axis=1)
FullDescKmeans12.columns = ['Cluster','Description','Neighbourhood']
g = sns.barplot(x=FullDescKmeans12['Cluster'].value_counts().index,y=FullDescKmeans12['Cluster'].value_counts()) 
plt.title("Number of listings in each cluster")


# ### Check cross-tab between cluster and neighbourhood

# Create crosstab between Cluster and Neighbourhood 
ctab = pd.crosstab(index=FullDescKmeans12['Neighbourhood'],columns=FullDescKmeans12['Cluster'])
plt.figure(figsize=(10,10))
sns.heatmap(ctab,annot=True,cmap='Blues', fmt='g')
plt.title("Crosstab of Cluster and Neighbourhood")


# ### Show top-30 words that best describe each listing cluster

# I previously use a regular print-out of the words, but now I am using a wordcloud instead
modelSegment = pipeline.fit(df['Descriptions_text'],FullDescKmeans12['Cluster'])
# show_topn(modelSegment.named_steps['CLF'], modelSegment.named_steps['tf_idf'], [str(i) for i in range(12)], 20)

# Create wordcloud based on top-50 words
Keywords12 = {}
save_topn(modelSegment.named_steps['CLF'], modelSegment.named_steps['tf_idf'], [str(i) for i in range(12)], 30,outdict=Keywords12)
fig,axes=plt.subplots(4,3,figsize=(20,20))
for i in range(12):
    wordlist = list(Keywords12[i])
    Word_Cloud = WordCloud(background_color='black',max_words=50,relative_scaling=0.2).generate(" ".join(wordlist))
    print(Word_Cloud)
    axes[math.floor(i/3),i%3].imshow(Word_Cloud)
