import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# import data
data_set = pd.read_csv('Mall_Customers.csv')
x = data_set[['Annual Income (k$)', 'Spending Score (1-100)']]


# choose the best number of clusters (K Value) using Elbow Method
wcss = []
for i in range(1, 11):
    model = KMeans(n_clusters=i)
    model.fit(x)
    wcss.append(model.inertia_)
plt.plot(range(1, 11), wcss)
plt.show()


# from the wcss , turns out the best clusters number is 5
model = KMeans(n_clusters=5)
y_labels = model.fit_predict(x)
# y_labels = model.labels_


# predicting a new value
new_value = model.predict([[15, 80]])
print('new value is from cluster: {}'.format(new_value))


# visualising the clusters
plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'], c=y_labels, cmap='rainbow')
centers = model.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], s=100, c='black')

