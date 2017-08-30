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



# LabelEncoder, OneHotEncoder ( Handling categorial values )
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
X['Country'] = label_encoder.fit_transform(X['Country'])
hot_encoder = OneHotEncoder(categorical_features=[0])
X = hot_encoder.fit_transform(X).toarray()

