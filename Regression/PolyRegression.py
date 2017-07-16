import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# init
size = np.array([i for i in range(10)]).reshape(-1, 1)
prices = np.array([i ** 2 for i in range(10)])

# model
model = make_pipeline(PolynomialFeatures(2), Ridge())
model.fit(size, prices)
print 'Accuracy is:', model.score(size, prices)

# predict
predicted_price = model.predict(10)
print 'Price for house with size %d = %d' % (10, predicted_price)


""" ---- Plotting All ---- """
# data set plot
plt.plot(size, prices, 'ro', label='Data-Set House Prices', linewidth=3)
# plot the approx function
plt.plot(size, model.predict(size), 'g-', label='Approx Function', linewidth=3)
plt.legend()
plt.xlabel('sizes')
plt.ylabel('prices')
plt.show()
