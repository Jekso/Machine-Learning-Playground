import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# init
size = np.array([i for i in range(10)]).reshape(-1, 1)
prices = np.array([i ** 2 for i in range(10)])

# model
model = LinearRegression()
model.fit(size, prices)
print 'Accuracy is:', model.score(size, prices)

# predict
# predicted_price = model.coef_*10+model.intercept_
predicted_price = model.predict(10)
print 'Price for house with size %d = %d' % (10, predicted_price)

# plotting all
# data set plot
plt.plot(size, prices, 'r.', label='house prices', linewidth=3)
# plot the predicted point
plt.plot(10, predicted_price, 'go', label='predicted house prices', linewidth=3)
# plot the approx function
plt.plot(size, model.predict(size), 'b-', label='approx function', linewidth=3)
plt.legend()
plt.xlabel('sizes')
plt.ylabel('prices')
plt.show()
