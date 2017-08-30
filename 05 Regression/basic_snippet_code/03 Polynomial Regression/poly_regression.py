import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# import data
data_set = pd.read_csv('Position_Salaries.csv')
x = data_set['Level']
y = data_set['Salary']


# split data to train & test sets
x_train, x_test, y_train, y_test = train_test_split(x, y)


# generate the polynomial features from x: x^0, x^1, x^2, ...
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))


# train the model using the ploynomial features
model = LinearRegression()
model.fit(x_train_poly, y_train)


# generate the ploy features from test set, then predict  
x_test_poly = poly.transform(x_test.reshape(-1, 1))
y_pred = model.predict(x_test_poly)


#plot the results
plt.plot(x_train, y_train, 'ro', label='training_data')
plt.plot(x_test, y_test, 'bo', label='testing_data')
plt.plot(x_test, y_pred, 'go', label='predicted_data')
plt.legend()
plt.show()