import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

dataset = pd.read_csv('../Machine Learning A-Z/Datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling of the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Implementation of the ML Algo
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predict the Resultant Value
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sbn.heatmap(cm, annot = True)

# Applying the k-fold Cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10) 
accuracies.mean()
accuracies.std()

# Implementing the Logistic Regression
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor_accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
regressor_accuracies.mean()
regressor_accuracies.std()

