
# coding: utf-8

# In[73]:

# import the fuckin modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


# In[74]:

# import the csv dataset
dataset = pd.read_csv("spam_dataset.csv", sep='\t', names=['Status', 'Messages'])


# In[75]:

# show the first 5 rows of the datase
dataset.head()


# In[76]:

# change categoral feature to numerical feature
dataset = pd.get_dummies(data=dataset,columns=['Status'], drop_first=True)


# In[77]:

# split the dataset to training & testing sets
X = dataset["Messages"]
Y = dataset["Status_spam"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=22)


# In[78]:

# initialize the tfidv object with english stopping words
tfidv = TfidfVectorizer(stop_words='english')


# In[79]:

# convert the text to frequency table
x_train = tfidv.fit_transform(x_train)
x_test = tfidv.transform(x_test)


# In[80]:

# start train the naive_bays model on the data
model = MultinomialNB()
model.fit(x_train, y_train)


# In[81]:

# test the model on the test_set
y_pred = model.predict(x_test)


# In[83]:

#calculate the accuracy
print("Accurcy =",accuracy_score(y_test, y_pred))


# In[84]:

#print confusion_matrix
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))


# In[ ]:



