from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# import iris data-set
data = datasets.load_iris()
x = data.data
y = data.target

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y)

# init the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# predict
y_predict = model.predict(x_test)

# test the accuracy & print the confusion matrix
print confusion_matrix(y_test, y_predict)
print 'Accuracy is: ', accuracy_score(y_test, y_predict)




