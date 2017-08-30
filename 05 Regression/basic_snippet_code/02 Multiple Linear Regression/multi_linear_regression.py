import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# import & clean data (handle categorial values)
data_set = pd.read_csv('50_Startups.csv')
data_set = pd.get_dummies(data=data_set, columns=['State'], drop_first=True)
x = data_set.drop('Profit', axis=1)
y = data_set['Profit']


# split the data to training & testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y)


# training the model
model = LinearRegression()
model.fit(x_train, y_train)


# predicting the test set
y_pred = model.predict(x_test)


# calculate the score 
print(model.score(x_test, y_test))