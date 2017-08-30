# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
data_set = pd.read_csv('Data.csv')

X = data_set.iloc[:,:-1].values
X = data_set.drop('Purchased', axis=1)

y = data_set.iloc[:,-1].values
y = data_set['Purchased']