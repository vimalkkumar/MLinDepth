# Mean Shift Clustering Algorithm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/Mall_Customers.csv') 
X = df.iloc[:, [3, 4]].values

from sklearn.cluster import MeanShift, estimate_bandwidth
# Estimate bandwith
bandwidth = estimate_bandwidth(X, quantile=0.15, random_state = 0)
# Applying the meanshift algorithm
meanshift_cluster = MeanShift(bandwidth = bandwidth)
y_meanshift = meanshift_cluster.fit_predict(X)
# Number of Clusters in the given datasets
labels = meanshift_cluster.labels_
n_clusters = np.unique(labels)
print('Optimal number of clusters = ', len(n_clusters))

# Visualing the K-Means Clustering Algo
plt.scatter(X[y_meanshift == 0, 0], X[y_meanshift == 0, 1], s = 100, c = 'red', label = 'Cluster 1 : Careful')
plt.scatter(X[y_meanshift == 1, 0], X[y_meanshift == 1, 1], s = 100, c = 'blue', label = 'Cluster 2 : Standard')
plt.scatter(X[y_meanshift == 2, 0], X[y_meanshift == 2, 1], s = 100, c = 'green', label = 'Cluster 3 : Target')
plt.scatter(X[y_meanshift == 3, 0], X[y_meanshift == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4 : Careless')
plt.scatter(X[y_meanshift == 4, 0], X[y_meanshift == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5 : Sensible')
plt.scatter(meanshift_cluster.cluster_centers_[:, 0], meanshift_cluster.cluster_centers_[:, 1], s = 50, c = 'black', label = 'Centroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Scores (1-100)')
plt.legend()
plt.show()