# Decision Tree Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/Position_Salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


#Visualising the Decision Tree Regression 
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Trouth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

# Visualising the Decision Tree Regression with High Resoluation
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Trouth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()