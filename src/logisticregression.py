#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Logistic Regression
import pandas as pd
import numpy as np
x = pd.read_csv("Desktop/jane-street-market-prediction/input_data.csv").to_numpy()
resp = pd.read_csv("Desktop/jane-street-market-prediction/input_data.csv").to_numpy()

##
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model as skl_lm
import numpy as np


pca = PCA(n_components = 1)
resp_pca = pca.fit_transform(resp)
y = (resp_pca > 0).astype("int")
y = np.ravel(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.70, random_state = 1)
print("Train size:", y_train.shape[0], "; % trues:", np.sum(y_train)/y_train.shape[0])
print("Test size:", y_test.shape[0], "; % trues:", np.sum(y_test)/y_test.shape[0])

# fit model
clf = skl_lm.LogisticRegression(solver='newton-cg') ## Command for Logistic Regression. We are using Newton's method for computing MLE. You are not required to know what it is exactly for the purpose of this course.
clf.fit(x_train,y_train)

# predictions on training set
yhat_train = clf.predict(x_train)
print("Train MSE:", accuracy_score(y_train, yhat_train))

# predictions on test set
yhat_test = clf.predict(x_test)
print("Test MSE:", accuracy_score(y_test, yhat_test))


# In[ ]:


#Train size: 499356 ; % trues: 0.30018063265485945
#Test size: 1165164 ; % trues: 0.29904888925507483
#Train MSE: 0.9997256466328631
#Test MSE: 0.9996283784943579

