import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


# import data
data_set = pd.read_csv('Position_Salaries.csv')
x = data_set['Level']
y = data_set['Salary']


# split data to train & test sets
x_train, x_test, y_train, y_test = train_test_split(x, y)


# pipeline model
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(x_train.reshape(-1, 1), y_train)
y_pred = model.predict(x_test.reshape(-1, 1))


#plot the results
plt.plot(x_train, y_train, 'ro', label='training_data')
plt.plot(x_test, y_test, 'bo', label='testing_data')
plt.plot(x_test, y_pred, 'go', label='predicted_data')
plt.legend()
plt.show()
