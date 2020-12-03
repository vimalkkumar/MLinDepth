# Apriori Algorith - Association Rule Learning
import numpy as np
import pandas as pd

df = pd.read_csv('Datasets/store_data.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(df.values[i, j]) for j in range(0, 20)])

# Training Apriori Model on the datasets
from Library.apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

#reset -f