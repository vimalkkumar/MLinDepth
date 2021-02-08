# SVR (Support Vector Regression)
# Importing the important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/Position_Salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values
y = np.reshape(y, (10, 1))

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = np.array(y).reshape(-1, 1)
y = sc_y.fit_transform(y)
#y = np.squeeze(sc_y.fit_transform(y.reshape(-1, 1)))

# Fitting SVR to the Dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y) 

# Visualising the SVR Result 
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('SVR(Trouth or Bluff)')
plt.xlabel('Position or Level')
plt.ylabel('Salary')
plt.show()

# Predicting the new Result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
