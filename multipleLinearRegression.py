import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/50_Startups.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, 4].values

# Encoding the Categorical Data
from sklearn.preprocessing import OneHotEncoder
#OneHotEncoder Using ColumnTransformer Class
from sklearn.compose import ColumnTransformer
columnTransform = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   # The column numbers to be transformed (here is [3] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X = columnTransform.fit_transform(X)
X = np.array(X, dtype='float')

# Avoid the Dummy Variable Trap
X = X[:, 1:]
#X = pd.DataFrame(X)

# Now its time to split the data into Training and Test Datasets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting the Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test Set Result 
Y_pred = regressor.predict(X_test)

# Building the optimal model using the backward elimination method
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()   # Ordinary Least Squares
regressor_OLS.summary()

X_opt = X[:, [0, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()