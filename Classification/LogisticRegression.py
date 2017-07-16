import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# init
x = np.array([i for i in range(50)]).reshape(-1, 1)
y = np.array([5 for i in range(25)]+[10 for i in range(25, 50)])

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y)

# plotting
plt.plot(x_train, y_train, 'go', label='data points', linewidth=5)
plt.xlabel('x values')
plt.ylabel('y values')
plt.legend()
plt.show()

# fitting the model with training data
model = LogisticRegression()
model.fit(x_train, y_train)

# predict and accuracy
print 'Test data-set values:', x_test
print 'Predicted values:', model.predict(x_test)
print 'Accuracy:', model.score(x_test, y_test)
print 'Probability of each predicted values: ', model.predict_proba(x_test)