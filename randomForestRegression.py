import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/Position_Salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

# Fitting the Random Forest Regression in to the Dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)  # estimators 10, 100, 300, 500
regressor.fit(X, y)

# Visualising the Random Forest Regression
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Trouth or Bluf (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting the result
y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))