#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  


# In[ ]:


import pandas as pd

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


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)  


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


# In[ ]:


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 20):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(12, 6))  
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  

