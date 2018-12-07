
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




df1 = listings[["number_of_reviews", "review_scores_rating","review_scores_accuracy","review_scores_communication","review_scores_location", "review_scores_cleanliness","review_scores_checkin","review_scores_value","cancellation_policy", 
               "reviews_per_month","neighbourhood_cleansed"]]
df1.head()


df = df1.dropna(axis = 0,how= 'any') 
df.head()




df_c = df[["review_scores_accuracy", "review_scores_communication",
           "review_scores_location", "review_scores_cleanliness", "review_scores_checkin", "review_scores_value"]]

cm = np.corrcoef(df_c.T)
sns.heatmap(cm, annot=True, yticklabels=df_c.columns, xticklabels=df_c.columns)


# # prediction examples


# split into test and training data
np.random.seed(1)
indices = np.random.permutation(len(df))
train_size = int(round(0.8*len(df)))
test_size = len(df)-train_size

    
y = df['review_scores_rating']
x = df[["review_scores_accuracy","review_scores_communication",
        "review_scores_location", "review_scores_cleanliness","review_scores_checkin","review_scores_value"]]

x.train = x.iloc[indices[0:train_size]]
y.train = y.iloc[indices[0:train_size]]
x.test = x.iloc[indices[train_size+1:]]
y.test = y.iloc[indices[train_size+1:]]

x2 = x.train.as_matrix()
y2 = y.train.as_matrix()



x0 = x.test.as_matrix()
y0 = y.test.as_matrix()



# different method
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=1)
#slr = linear_model.LinearRegression()
slr = linear_model.LogisticRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)




import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_test_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Linear Regression RMSE: %.4f"
      % mean_squared_error(y_test, y_test_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_test_pred))




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
plt.ylabel('True Value')
plt.legend(loc='upper left')
plt.show()



from sklearn.metrics import r2_score, f1_score, recall_score, accuracy_score
from sklearn.metrics import mean_squared_error,mean_squared_log_error
rms_ols2=sqrt(mean_squared_error(y_test,y_test_pred))
print('MSE train: %.3f, test: %.3f' % (
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

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))



plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.show()




rmse_randfor=sqrt(mean_squared_error(y_test,y_test_pred))
print(rmse_randfor)



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





from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
model = ensemble.GradientBoostingRegressor()
model.fit(X_train, y_train)





print('Random Forest R squared": %.4f' % model.score(X_test, y_test))



y_pred = model.predict(X_test)
model_mse = mean_squared_error(y_pred, y_test)
model_rmse = np.sqrt(model_mse)
print('Gradient Boosting RMSE: %.4f' % model_rmse)




