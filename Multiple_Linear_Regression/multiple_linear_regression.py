# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

print(X.shape, y.shape)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy variable trap
X = X[:,1:]   #remove index column (not necessary as algorithm takes care of it)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_Test = train_test_split(X,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)




#Building optimal model using Backward Elimination

import statsmodels.formula.api as sm 

X = np.append(arr = np.ones((50,1)).astype(int),values = X,axis = 1)
X_opt = X[:,:]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
#print(regressor_OLS.summary())

# Now remove indepedent variable X2 as it has the highest P-Value
#The lesser the P value the more its significance

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
#print(regressor_OLS.summary())


# Now remove indepedent variable X1 as it has the highest P-Value

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
#print(regressor_OLS.summary())

# Now remove indepedent variable X2(i.e 4th column of original dataset) as it has the highest P-Value

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())

# Now remove indepedent variable X2(i.e 5th column of original dataset) as it has the highest P-Value

'''X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())'''


#Prediction

X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_opt, y, test_size = 1/3, random_state = 0)
regressor_opt = LinearRegression()
regressor_opt.fit(X_opt_train, y_opt_train)
 
y_opt_pred = regressor_opt.predict(X_opt_test)

