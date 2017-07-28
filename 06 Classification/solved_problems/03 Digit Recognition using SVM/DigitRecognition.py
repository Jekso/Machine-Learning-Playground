from sklearn import svm
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# load data and prepare the training set
data = datasets.load_digits()
x = np.array(data.data[:-5])
y = data.target[:-5]

# fitting the model - gamma (learning rate) is very small to maximize accuracy, we could use 0.001
model = svm.SVC(gamma=0.0001, C=100)
model.fit(x, y)

# predict the digit
print model.predict([data.data[-2]])

# plot the digit
plt.imshow(data.images[-2])
plt.show()

