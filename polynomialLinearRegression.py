# Polynomial Linear Regression
# Importing the important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/Position_Salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

# Fitting the Linear Regression to the Datasets
from sklearn.linear_model import LinearRegression
linRegressor = LinearRegression()
linRegressor.fit(X, y)

# Fitting the Polynomiaol Regression in to the Datasets
from sklearn.preprocessing import PolynomialFeatures
polyRegressor = PolynomialFeatures(degree = 4)   # Change the degree 2 to 3, 4, 5 see the fine graph
polyX = polyRegressor.fit_transform(X)

linRegressor_2 = LinearRegression()
linRegressor_2.fit(polyX, y)

# Visualising the Linear Regression
plt.grid()
plt.scatter(X, y, color = 'red')
plt.plot(X, linRegressor.predict(X), color = 'blue')
plt.title('Linear Regression Model (Truth OR Bluff)')
plt.xlabel('Level or Position')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial regression
plt.grid()
plt.scatter(X, y, color = 'red')
plt.plot(X, linRegressor_2.predict(polyRegressor.fit_transform(X)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Level or Position')
plt.ylabel('Salary')
plt.show()

# Predicting the new result by Linear Regression Model
linRegressor.predict(np.array([6.5]).reshape(1, 1))

# Predicting the new result by Polynomial Regression Model
linRegressor_2.predict(polyRegressor.fit_transform([[6.5]]))


