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
print(confusion_matrix(y_test, y_pred))
print(model.score(x_test, y_test))


# accuracy using K-Fold Cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=model, X=x_train, y=y_train, cv=10, verbose=1)
print(accuracies.mean())


# getting the best model params using GridSearchCV
from sklearn.model_selection import GridSearchCV
params = [
        {'C':[1, 10, 100], 'kernel':['linear']},
        {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.5, 0.6, 0.7, 0.1,0.01,0.01]}
         ]
grid_search = GridSearchCV(estimator=model,
                           param_grid=params,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1,
                           verbose=3)
grid_search.fit(x_train, y_train)
best_score = grid_search.best_score_
best_params = grid_search.best_params_