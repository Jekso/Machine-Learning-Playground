import numpy as np
from sklearn import svm

x = np.array([[100, 150], [3, 2], [10, 25], [200, 200], [15, 10]])
y = np.array([2, 1, 1, 2, 1])

model = svm.SVC()
model.fit(x, y)
print model.predict(np.array([[500, 70]]))

