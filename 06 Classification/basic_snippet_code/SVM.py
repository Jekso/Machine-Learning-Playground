import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# import & clean data
data_set = pd.read_csv('Social_Network_Ads.csv')
data_set = pd.get_dummies(data=data_set, columns=['Gender'], drop_first=True)
x = data_set[['Age', 'EstimatedSalary', 'Gender_Male']]
y = data_set['Purchased']


# scale the data
scaler = StandardScaler()
x = scaler.fit_transform(x)


# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y)


# fitting the model with training data
model = SVC()
model.fit(x_train, y_train)

# predict and accuracy
y_pred = model.predict(x_test)
# print('Probability of each predicted values: {}'.format(model.predict_proba(x_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy: {}'.format(model.score(x_test, y_test)))
print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
