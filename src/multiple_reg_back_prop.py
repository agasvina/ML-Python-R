# Multiple Linear Regression. 
# y = b + b1 * x1 + b2 + x2 + ....
# We are going to use the backward propagation here. 

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Library for oneHotEncoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Library for spiltting the Train and test set:
from sklearn.cross_validation import train_test_split

#Several libraries for the linear regression
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


dataset = pd.read_csv('/Users/luca/git/ML-Python/data/50_Startups.csv')
#The dependent variable
X = dataset.iloc[:, :-1].values
#The independent variable:
y = dataset.iloc[:, 4].values

#Encoding the categorical data 
X[:,3] = LabelEncoder().fit_transform(X[:,3])
oneHotEncoder = OneHotEncoder(categorical_features =[3])
X = oneHotEncoder.fit_transform(X).toarray()

#To Avoid the Dummy variable trap, we remove one of the dummy variable coloumn
#Why? (Check later)
X = X[:,1:]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Doing  the Multiple Regression without back propagation:
#Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Using the backward propagation:
#Adding constant to the dependent variable:
X = np.append(arr=np.ones((50,1)).astype(int), values= X, axis = 1)
indexes = [0,1,2,3,4,5]
X_opt = X[:, indexes]
regressor_OLS = None;
mask = np.array([9999])


#Creating the loop:
while np.count_nonzero(mask) > 0:
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    print(regressor_OLS.summary())

    mask = regressor_OLS.pvalues > 0.05
    counter = np.count_nonzero(mask) > 0
    toBeDeleted = np.where(regressor_OLS.pvalues == regressor_OLS.pvalues.max())
    if counter > 0:
         indexes.remove(indexes[toBeDeleted[0].tolist()[0]])
         X_opt = X[:, indexes]

    #For to break if too many variables is been deleted
    if(len(indexes) < 3):
        break

print("".join(("Relevant Dependant variable: ", str(indexes))))












