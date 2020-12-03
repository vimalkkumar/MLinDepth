# Importing the Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset

df = pd.read_csv("Datasets/Data.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, 3].values

# It's time to take care of Missing Data Older Version

#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) 
#imputer = imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])

####### SIMPLE IMPUTER #######

from sklearn.impute import SimpleImputer
# Using MEAN
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
#print(imputer.transform(X[:, 1:3]))

# Using MEDIAN
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Using MOST_FREQUENT
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Using CONSTANT
imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = None)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
 
###### ITERATIVE IMPUTER ######
#from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(random_state = 0)
impuetr = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

###### ENCODING CATEGORICAL DATA ##### 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding the data
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
# Dummy Encodig for the Categorical Data
oneHotEncoder = OneHotEncoder(categories='auto')
X = oneHotEncoder.fit_transform(df[['Country']]).toarray()

#from sklearn.compose import ColumnTransformer
#columnTransform = ColumnTransformer(
#    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   # The column numbers to be transformed (here is [3] but can be [0, 1, 3])
#    remainder='passthrough'                                         # Leave the rest of the columns untouched
#)
#X = columnTransform.fit_transform(X)

gtdm_X = pd.get_dummies(X)
oneHotEncoder = OneHotEncoder(categories = [0])
X = oneHotEncoder.fit_transform(X).toarray()

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

######## SPLITTING THE DATA INTO TRAINING SET AND TEST SET ########
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

########## FEATURE SCALING ##############
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
