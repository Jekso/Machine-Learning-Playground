import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv('data.csv')
# print data.head()
# print data.describe()
# print data.corr()
x = np.array(data[['income', 'age']])
y = np.array(data['default'])

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = LogisticRegression()
model.fit(x_train, y_train)
y_predected = model.predict(x_test)
print confusion_matrix(y_test, y_predected)
print accuracy_score(y_test, y_predected)

