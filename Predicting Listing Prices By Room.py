

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels
import sklearn
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')




listings = pd.read_csv('Dublin_Airbnb_listings.csv')




listings.columns



df = listings[["host_response_rate", "host_acceptance_rate", "host_is_superhost",
               "host_listings_count", "zipcode", "property_type","room_type", "accommodates", "bathrooms", "bedrooms", 
               "beds", "price", "number_of_reviews", "review_scores_rating", "cancellation_policy", 
               "reviews_per_month", "neighbourhood_cleansed"]]

df.head()

print(df[['bathrooms','bedrooms','beds']].describe())



import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
X = df[['bathrooms','bedrooms','beds']]

from pandas.plotting import table

fig, ax = plt.subplots(1, 1)
table(ax, np.round(X.describe(), 2), loc='upper right', colWidths=[0.2, 0.2, 0.2])

X.plot.hist(ax=ax, stacked=True, bins=20, figsize = [8,8] )
ax.legend(loc='lower right', frameon=False)
plt.show()




df_c = df[['bathrooms','bedrooms','beds']]
      
cm = np.corrcoef(df_c.T)
sns.heatmap(cm, annot=True, yticklabels=df_c.columns, xticklabels=df_c.columns)




np.round(X.describe(), 2)




color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
X.plot.box(figsize = [8,8], color=color, sym='r+')



# split into test and training data
np.random.seed(1)
indices = np.random.permutation(len(df))
train_size = int(round(0.8*len(df)))
test_size = len(df)-train_size

y = df['price']
x = df[["bathrooms", "bedrooms","beds"]]

x.train = x.iloc[indices[0:train_size]]
y.train = y.iloc[indices[0:train_size]]
x.test = x.iloc[indices[train_size:]]
y.test = y.iloc[indices[train_size:]]

x2 = x.train.as_matrix()
y2 = y.train.as_matrix()




import statsmodels.api as sm
olsmod = sm.OLS(y2,x2)
olsres = olsmod.fit()
print(olsres.summary())




x0 = x.test.as_matrix()
y0 = y.test.as_matrix()




ypred = olsres.predict(x0) # out of sample prediction
from sklearn.metrics import mean_squared_error
from math import sqrt
rms_ols = sqrt(mean_squared_error(y0,ypred))




rms_ols



# plot predictions agains true values as function of bedrooms
beds = x.test['bedrooms'].as_matrix()




# different method
from sklearn import linear_model
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1)
slr = linear_model.LinearRegression()
#slr = linear_model.LogisticRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)



plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.show()



plt.scatter(y_train_pred, y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test,
            c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('actual value')
plt.legend(loc='upper left')
plt.show()

print('Coefficients: \n', slr.coef_)


# In[52]:


from sklearn.metrics import r2_score, f1_score, recall_score, accuracy_score
from sklearn.metrics import mean_squared_error,mean_squared_log_error
rms_ols2=sqrt(mean_squared_log_error(y_test,y_test_pred))
print('Mean Squared Log Error train: %.3f, test: %.3f' % (
        mean_squared_log_error(y_train, y_train_pred),
        mean_squared_log_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# # random forest


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=500, 
                               criterion='mse', 
                               random_state=3, 
                               n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('mean squared log error train: %.3f, test: %.3f' % (
        mean_squared_log_error(y_train, y_train_pred),
        mean_squared_log_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))



plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='orange', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.show()




rmse_randfor=sqrt(mean_squared_log_error(y_test,y_test_pred))
print(rmse_randfor)


# ### feature importance


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


importance = forest.feature_importances_
importance = pd.DataFrame(importance, index=x.columns, 
                          columns=["Importance"])

importance["Std"] = np.std([forest.feature_importances_
                            for tree in forest.estimators_], axis=0)

print(importance)







