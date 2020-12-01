# Kernel Support Vector Machine for Non-linearly Separable Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/Social_Network_Ads.csv')
X = df.iloc[:, 2:4].values
y = df.iloc[:, 4].values

# Split the Dataset into Train and Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Now Its time to impliment the SVM Classifier
from sklearn.svm import SVC
classifier_rbf = SVC(kernel = 'rbf', random_state = 0)  # Changing the kernel value fron linear to rbf (Gaussion)
classifier_rbf.fit(X_train, y_train)

# Predict the Resultant Value
y_pred_rbf = classifier_rbf.predict(X_test)

# Classifier for Linear kernel
classifier_linear = SVC(kernel = 'linear', random_state = 0)  # Changing the kernel value fron linear to rbf (Gaussion)
classifier_linear.fit(X_train, y_train)

# Predict the Resultant Value
y_pred_linear = classifier_linear.predict(X_test)

# Classifier for SIGMOID kernel
classifier_sigmoid = SVC(kernel = 'sigmoid', random_state = 0)  # Changing the kernel value fron linear to rbf (Gaussion)
classifier_sigmoid.fit(X_train, y_train)

# Predict the Resultant Value
y_pred_sigmoid = classifier_sigmoid.predict(X_test)

# Classifier for SIGMOID kernel
classifier_poly = SVC(kernel = 'poly', random_state = 0)  # Changing the kernel value fron linear to rbf (Gaussion)
classifier_poly.fit(X_train, y_train)

# Predict the Resultant Value
y_pred_sigmoid = classifier_sigmoid.predict(X_test)

# Checking the accuracy of the Traning Set
print('RBF Train : ', classifier_rbf.score(X_train, y_train))
print('RBF Test : ', classifier_rbf.score(X_test, y_test))
print('Linear Train : ', classifier_linear.score(X_train, y_train))
print('Linear Test : ', classifier_linear.score(X_test, y_test))
print('Sigmoid Train : ', classifier_sigmoid.score(X_train, y_train))
print('Sigmoid Test : ', classifier_sigmoid.score(X_test, y_test))
print('Poly Train : ', classifier_poly.score(X_train, y_train))
print('Poly Test : ', classifier_poly.score(X_test, y_test))

# Lest Understand the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_rbf)
print(cm)

# Visualising the Training Set Result
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_rbf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Support Vector Machine (Traning Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test Set Result
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier_rbf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Support Vector Machine (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()