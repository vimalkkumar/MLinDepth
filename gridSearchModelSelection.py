import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Datasets/Social_Network_Ads.csv')
X = df.iloc[:, [2, 3]].values
y = df.iloc[:, -1].values

# Spliting the dataset into training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Features Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting the Kernal SVM to the training dataset
from sklearn.svm import SVC
svc_classifier = SVC(kernel = 'rbf', random_state = 0)
svc_classifier.fit(X_train, y_train)

# Predicting the result
y_pred = svc_classifier.predict(X_test)

# Checking the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying the k-fold Cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = svc_classifier, X = X_train, y = y_train, cv = 10) 
accuracies.mean()
accuracies.std()

# Applying the grid search to find the best model and the best parameters
# Parameter Tunning
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [1, 10, 100, 1000], 
               'kernel' : ['linear']},
              {'C' : [1, 10, 100, 1000], 
               'kernel' : ['rbf', 'sigmoid'], 
               'gamma' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.7, 0.75 ,0.8, 0.9, 1.0]}]
grid_search = GridSearchCV(estimator = svc_classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameter = grid_search.best_params_
