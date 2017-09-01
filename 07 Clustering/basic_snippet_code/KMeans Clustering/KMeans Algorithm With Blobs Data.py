import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

x, y = make_blobs()


model = KMeans(6)
model.fit(x, y)
y_predicted = model.predict(x)
