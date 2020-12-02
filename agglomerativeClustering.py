# Hierarchical Clustering - Agglomerative Clustring
# reset -f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/Mall_Customers.csv')
X = df.iloc[:, [3, 4]].values

# Apply the Dendrogram for selecting the optimal number of Cluster
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram For Mall Customers')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Its time to apply Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_cluster = cluster.fit_predict(X)

# Visualising the Agglomerative Cluster
plt.scatter(X[y_cluster == 0, 0], X[y_cluster == 0, 1], s = 100, c = 'red', label = 'Cluster 1 : Careful') 
plt.scatter(X[y_cluster == 1, 0], X[y_cluster == 1, 1], s = 100, c = 'blue', label = 'Cluster 2 : Standard') 
plt.scatter(X[y_cluster == 2, 0], X[y_cluster == 2, 1], s = 100, c = 'green', label = 'Cluster 3 : Target') 
plt.scatter(X[y_cluster == 3, 0], X[y_cluster == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4 : Careless') 
plt.scatter(X[y_cluster == 4, 0], X[y_cluster == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5 : Sensible') 
plt.title('Clusters of Customers')
plt.xlabel('Annual Encome (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
