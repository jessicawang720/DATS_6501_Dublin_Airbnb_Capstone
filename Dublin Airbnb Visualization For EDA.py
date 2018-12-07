
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

listings = pd.read_csv('Dublin_Airbnb_listings.csv')

listings.head()

# # Different Visualizations for EDA

listings.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(10,7),
    c="price", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)

plt.figure(figsize = (10, 10))
sns.boxplot(x = 'neighbourhood_cleansed', y = 'price',  data = listings)
xt = plt.xticks(rotation=90)

sns.violinplot('neighbourhood_cleansed', 'price', data = listings)
xt = plt.xticks(rotation=90)

sns.factorplot('neighbourhood_cleansed', 'price', data = listings, color = 'm',                estimator = np.median, size = 4.5,  aspect=1.35)
xt = plt.xticks(rotation=90)

#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
listings.hist(bins=50, figsize=(20,15))
plt.savefig("attribute_histogram_plots")
plt.show()


plt.figure(figsize=(10,10))
sns.heatmap(listings.groupby([
        'neighbourhood_cleansed', 'bedrooms']).price.mean().unstack(),annot=True, fmt=".0f")

plt.figure(figsize=(10,10))
sns.heatmap(listings.groupby(['property_type', 'bedrooms']).price.mean().unstack(), annot=True, fmt=".0f", cmap="Blues")


plt.figure(figsize=(10,10))
sns.heatmap(listings.groupby(['beds', 'bedrooms']).price.mean().unstack(), annot=True, fmt=".0f")

listings['price'].hist(bins=50)
plt.ylabel('Count')
plt.xlabel('Listing price in $')
plt.title('Airbnb for the listing prices')


plt.scatter(listings['price'],listings['bedrooms'])
plt.ylabel('bedrooms')
plt.xlabel('Listing price in $')
plt.title('Number of bedrooms vs Price')

plt.scatter(listings['number_of_reviews'],listings['price'])
plt.ylabel('Listing price in $')
plt.xlabel('No. of reviews')
plt.title('Number of reviews vs price')

listings.pivot(columns = 'bedrooms',values = 'price').plot.hist(stacked = True,bins=25,figsize = (12, 8))
plt.xlabel('Listing price in $')

listings.pivot(columns = 'accommodates',values = 'price').plot.hist(stacked = True,bins=25,figsize = (12, 8))
plt.xlabel('Listing price in $')

listings.pivot(columns = 'room_type', values = 'price').plot.hist(stacked = True, bins=25,figsize = (12, 8))
plt.xlabel('Listing price in $')
