#!/usr/bin/env python
# coding: utf-8

# In[30]:

import nltk
nltk.download('punkt')
import pandas as pd
df = pd.read_csv('Dublin_Airbnb_listings.csv')
data = df['summary'].values.tolist()

NO_DOCUMENTS = len(data)
print(NO_DOCUMENTS)
print(data[:5])


# In[31]:


#!pip install gensim
import re
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import similarities

#Load stopwords
stop_words = set(stopwords.words("english"))
#Lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def clean(review) :
    cleaned_review = re.sub('[^a-zA-Z]', ' ', str(review)) # Remove punctuation and any words not starting with alphabet
    cleaned_review = cleaned_review.lower() # make capitalized words lower cases
    words = word_tokenize(cleaned_review) # Tokenization
    words = [word for word in words if not word in stop_words] # Remove stop words
    words = [wordnet_lemmatizer.lemmatize(word) for word in words] #Lemmatize words
    return words

cleaned_data = []
for info in data:
    cleaned_data.append(clean(info))
# Create a Dictionary associate word to id
dict = corpora.Dictionary(cleaned_data)

# Transform texts to numeric
corpus = [dict.doc2bow(i) for i in cleaned_data]

# Have a look at how the 100th document looks like: [(word_id, count)]
print(corpus[100])


# In[32]:


# Define the number of topics
topic = 15

# Build the LDA model
lda = models.LdaModel(corpus=corpus, num_topics=topic, id2word=dict)

print('LDA model')
for index in range(0,topic):
    # top 8 topics
    print("Topic Number %s:" % str(index+1), lda.print_topic(index, 8))
print("-" * 150)

# Build the LSI model
lsi = models.LsiModel(corpus=corpus, num_topics=topic, id2word=dict)

print("LSI model")
for index in range(0,topic):
    # top 8 topics
    print("Topic Number %s:" % str(index+1), lsi.print_topic(index, 8))
print("-" * 150)


# In[42]:


# randomly pick one reviews to predict similarity.
reviews = pd.read_csv('Dublin_reviews.csv')
text = reviews.loc[40,'comments']
print(text)


# In[43]:


# I choose to compare LDA model and LSI model to predict similarity.
lda_i = similarities.MatrixSimilarity(lda[corpus])

bow = dict.doc2bow(clean(text))
# Let's perform some queries
similar_lda = lda_i[lda[bow]]

# Sort the similarities
similar_LDA = sorted(enumerate(similar_lda), key=lambda item: -item[1])

# Top most similar documents:
print(similar_LDA[:10])

# Let's see what's the most similar document
doc_id, similarity = similar_LDA[1]
print(data[doc_id][:1000])


# In[44]:


# Do the same similarity queries by using LSI model
lsi_i = similarities.MatrixSimilarity(lsi[corpus])
similar_lsi = lsi_i[lsi[bow]]
similar_LSI = sorted(enumerate(similar_lsi), key=lambda item:-item[1])
print(similar_LSI[:10])
doc_id, similarity = similar_LSI[1]
print(data[doc_id][:1000])


# In[ ]:





# In[29]:





# In[ ]:




