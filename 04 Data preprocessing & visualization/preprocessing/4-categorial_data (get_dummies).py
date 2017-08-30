# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
data_set = pd.read_csv('Data.csv')
X = data_set.drop('Purchased', axis=1)
y = data_set['Purchased']


# Imputer ( Handling missing values )
from sklearn.preprocessing import Imputer
imputer = Imputer()
X[['Age', 'Salary']] = imputer.fit_transform(X[['Age', 'Salary']])


# for handling categorial values, use get_dummies , it's easier :D 
X = pd.get_dummies(data=X, columns=['Country'], drop_first=True)
print(X)