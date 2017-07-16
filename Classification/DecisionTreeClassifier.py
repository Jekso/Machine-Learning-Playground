import numpy as np
from sklearn.tree import DecisionTreeClassifier

x = np.array([[100, 150], [3, 2], [10, 25], [200, 200], [15, 10]])
y = np.array([2, 1, 1, 2, 1])

model = DecisionTreeClassifier()
model.fit(x, y)
print model.predict(np.array([[150, 150]]))
