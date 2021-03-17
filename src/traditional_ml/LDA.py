#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Logistic Regression
import pandas as pd
import numpy as np
x = pd.read_csv("Desktop/jane-street-market-prediction/input_data.csv").to_numpy()
resp = pd.read_csv("Desktop/jane-street-market-prediction/output_data.csv")

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


# In[ ]:


X = x_test.as_matrix()
y = y_test.as_matrix()

lda = LinearDiscriminantAnalysis(solver='svd') ## SVD is another way of computing the LDA classification rule.
y_pred = lda.fit(X, y).predict(X)

df_ = pd.DataFrame({'True default status': y,
                    'Predicted default status': y_pred})
df_.replace(to_replace={0:'No', 1:'Yes'}, inplace=True)

df_.groupby(['Predicted default status','True default status']).size().unstack('True default status')


# In[ ]:


print(classification_report(y, y_pred, target_names=['No', 'Yes']))

