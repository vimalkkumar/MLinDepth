# Ridge and Lasso Regression 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.datasets import load_boston
dataset = load_boston()
boston_dataset = pd.DataFrame(dataset.data)

boston_dataset.columns = dataset.feature_names
boston_dataset['Price'] = dataset.target
X = boston_dataset.iloc[:, :-1].values
y = boston_dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
regressor = LinearRegression()
mse = cross_val_score(regressor, X, y, scoring = 'neg_mean_squared_error', cv = 10)
mse_mean = np.mean(mse)

# Implementation Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-5, 1e-3, 1e-2, 1, 5, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 100]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring = 'neg_mean_squared_error', cv = 10)
ridge_regressor.fit(X, y)

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

# Implementation Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-5, 1e-3, 1e-2, 1, 5, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 100]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring = 'neg_mean_squared_error', cv = 10)
lasso_regressor.fit(X, y)

print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

# Train Test Spilit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

lasso_prediction = lasso_regressor.predict(X_test)
ridge_prediction = ridge_regressor.predict(X_test)

sbn.distplot(y_test - lasso_prediction)
sbn.distplot(y_test - ridge_prediction)