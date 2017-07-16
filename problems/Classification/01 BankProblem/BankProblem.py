import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([[10000, 80000, 35], [7000, 120000, 57], [100, 23000, 22], [223, 18000, 26]]).reshape(-1, 3)
y = np.array([1, 1, 0, 0])

print x
print y

model = LogisticRegression()
model.fit(x, y)
new = np.array([5550, 50000, 25]).reshape(1, -1)
print new
print model.predict(new)
