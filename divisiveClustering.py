# Hierarchical Clustering - Divisive Clustering

# reset -f

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Datasets/Mall_Customers.csv')
x = df.iloc[:, [3, 4]].values


