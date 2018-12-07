# DATS_6501_Dublin_Airbnb


Data link: https://storage.googleapis.com/capstone_airbnb/Dataset.zip

##  Airbnb Dublin DATASET PROJECT
### Author: Tianyi Wang

In this project, the Airbnb in Dublin, Ireland will be focused. Ireland is an island country, tourism is one of the most important economic industry of Ireland, which expected will reach 6.0% of total GDP which is EUR 17.2 billion (USD 19.4 billion) in 2018. In 2017, it was 5.9% of GDP in 2017 and is forecast to rise by 4.2% in 2018, and to rise by 3.8% to EUR26.0 billion (USD29.4 billion), 7.1% of GDP in 2028. (Irish Tourism Industry Confederation). That’s the reason why I want to choose Dublin as my target city for my research. In Ireland, most of the attractions for tourists are at Dublin, especially during the St. Patrick’s Day every year, March 17th, the most tourists of the year. Because of the popularity for traveling, lots of local people willing to post their personal properties on Airbnb, also the demand of Airbnb is relatively high in Dublin. In my project, I'm going to use different kinds of Machine learning tools and Natural Language Processing to discover customer motives in choosing to stay with Airbnb and price of the accommodation.


There have two main parts of my project, the first part is using different machine learning tools such as linear regression, logistic regression, random forest to modeling the price and reviews rating. The second part is using Natural Language Processing based on comments from guests and description by the household. I did cluster descriptions of the room by the house owners.  By using household's summary of the listing and the comments by the guest, I chose the topic first then did comments comparison in the Latent Dirichlet Allocation and Latent Semantic Indexing Model to predict similarity.



### Running the code

* Python 2.7 or above

Data link: https://storage.googleapis.com/capstone_airbnb/Dataset.zip



Before running the code, please make sure you have the following packages installed:  matplotlib, sklearn, numpy, gensim, wordcloud, nltk, statsmodel. 


* Run 'Dublin Airbnb Visualization For EDA.py' first. This file include different visualiztion for the data. It will give you a big picture of what's the data looks like. 
* Run 'Predicting Listing Prices By Room.py' file, which is the code of different ML tools i used for the predicting listing the price. 
* Run 'Predicting_Listing_Reviews_Rating.py' file, which is the code of ML for Reviews rating score.
* Run 'Dublin Airbnb Descriptions NLP Cluster.py' file, which is the code of NLP for cluster the room by house owner’s descriptions.
* Run 'Descriptions NLP to Specific Neighborhood in Dublin.py' file, which is the code of luster the room by house owner’s descriptions for specific Neighborhood at Dublin.
* Run 'LDA_and_LSI.py' file, which is the code of  LDA and LSI models for the	predict similarity.






## Credit
This project is a collaborative effort of Tianyi Wang(jessicawang@gwu.edu),
