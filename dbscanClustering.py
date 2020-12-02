# DBSCAN (Density Based Spatial Clustering of Application with Noise) Algorithm
#reset -f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/Mall_Customers.csv')
X = df.iloc[:, [3, 4]].values

#
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors = 2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
distances = np.sort(distances, axis = 0)
distances = distances[:, 1]
plt.plot(distances)

# Features Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Implemantation of DBSCAN
from sklearn.cluster import DBSCAN
dbscan_cluster = DBSCAN(eps = 0.5, metric = 'euclidean', min_samples = 4)
y_dbscan = dbscan_cluster.fit(X)

# 
labels = dbscan_cluster.labels_
# identifying the core samples
from sklearn import metrics
core_samples = np.zeros_like(labels, dtype = bool)
core_samples[dbscan_cluster.core_sample_indices_] = True
print(core_samples)

# Calculating the number of Clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print('Optimal Number of Clusters : ', n_clusters)

# Computing the Silhouette Score
print('The Silhouette Score : %0.3f' % metrics.silhouette_score(X, labels))

# Visualing the DBSCAN Clustering Algo
colors =  ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
plt.scatter(X[:, 0], X[:, 1], c = vectorizer(labels))
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customers Culsters for Mall')
plt.show()