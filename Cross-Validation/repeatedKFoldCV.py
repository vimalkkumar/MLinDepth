import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

dataset = pd.read_csv('../Machine Learning A-Z/Datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

# Describing the whole dataset like Count, Mean Std, Min, Max 
details = dataset.describe()

# How many Data-points for each class are presents?
dataset['Purchased'].value_counts()

# Splitting the dataset into train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling of the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Implementating the Linear Regression Algorithm
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)

# Calculating the score of the Linear Regression Model
lr_score = lr_classifier.score(X_test, y_test)

# Implementation of the Support Vector Machine Algorithm
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'rbf', random_state = 0)
svm_classifier.fit(X_train, y_train)

# Calculating the score of the Support Vector Machine Model
svm_score = svm_classifier.score(X_test, y_test)

# Implementation of the Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(X_train, y_train)

# Calculating the score of the Decision Tree Model
dt_score = dt_classifier.score(X_test, y_test)

# Implementation of the Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 60, random_state = 0)
rf_classifier.fit(X_train, y_train)

# Calculating the score of the Random Forest Model
rf_score = rf_classifier.score(X_test, y_test)

# Implementation of the Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Calculating the score of the Random Forest Model
nb_score = rf_classifier.score(X_test, y_test)


from sklearn.model_selection import RepeatedKFold
"""
We have cross_val_score() to handle the above problem in sklearn library
"""
from sklearn.model_selection import cross_val_score, cross_val_predict
print('Avereage score of Logistic Regression : {}'\
      .format(cross_val_score(estimator = lr_classifier, X = X_test, y = y_test, cv = RepeatedKFold(n_splits = 10, n_repeats = 5)).mean()))
print('Avereage score of Support Vector Machine : {}'\
      .format(cross_val_score(SVC(), X_test, y_test, cv = RepeatedKFold(n_splits = 10, n_repeats = 5)).mean()))
print('Avereage score of Decision Tree : {}'\
      .format(cross_val_score(DecisionTreeClassifier(criterion = 'entropy'), X_test, y_test, cv = RepeatedKFold(n_splits = 10, n_repeats = 5)).mean()))
print('Avereage score of Random Forest : {}'\
      .format(cross_val_score(RandomForestClassifier(n_estimators = 60), X_test, y_test, cv = RepeatedKFold(n_splits = 10, n_repeats = 5)).mean()))
print('Avereage score of Naive Bayes : {}'\
      .format(cross_val_score(GaussianNB(), X_test, y_test, cv = RepeatedKFold(n_splits = 10, n_repeats = 5)).mean()))
