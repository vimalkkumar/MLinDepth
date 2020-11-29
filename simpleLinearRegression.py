
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DataFrame = pd.read_csv('Datasets/Salary_Data.csv')
X = DataFrame.iloc[:, :-1].values
y = DataFrame.iloc[:, 1].values

# Spliting the data into training and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting the Simple linear regression model into the traing set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Now its time to predict the salary on the basis of experiences
y_pred = regressor.predict(X_test)

# Time to Visualise the data
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experiences (For Training Set)')
plt.xlabel('Years of Experiences')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Salary vs Experiences (For Test Set)')
plt.xlabel('Years of Experiences')
plt.ylabel('Salary')
plt.show()
