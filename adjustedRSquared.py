import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/Persons_family_data.csv')
X = df.iloc[:, 1:4].values
y = df.iloc[:, 4].values

# Spliting the data in Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 3, random_state = 0)

# Performing the Linear Regression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# applying method of least squares
import statsmodels.api as sm
model = sm.OLS(endog = y, exog = X).fit()
model.summary()
predictions = model.predict(X_test)

# Its time to predict the result
prediction = regressor.predict(X_test)

# R Square of the Given Data
rsquared = regressor.score(X_test, y_test)

# Adjusted R Square of the given Data
n = len(df)             # Number of Records
p = len(df.columns)-2   #  number of features .i.e. columns excluding uniqueId and target variable
adjr = 1 - (1 - rsquared) * (n-1) / (n-p-1)

# Visualising the Regression Model
plt.scatter(X[:, 2], y_train, color = 'red')
plt.plot(X[:, 2], predition, color = 'blue')
plt.title("Person's family Data")
plt.xlabel('Person Weight')
plt.ylabel('Person Height')
plt.show()