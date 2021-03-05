#!/usr/bin/env python
# coding: utf-8

# In[261]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from sklearn import linear_model, datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import linear_model

import sklearn.linear_model as skl_lm
import matplotlib.pyplot as plt
import seaborn as sns
import rfpimp
import pandas as pd
import numpy as np
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from sklearn import linear_model, datasets
import seaborn as sns
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt


import seaborn as sns
import itertools


# In[14]:


df = pd.read_csv("Desktop/jane-street-market-prediction/train.csv")
x = pd.read_csv("Desktop/jane-street-market-prediction/input_data.csv").to_numpy()
resp = pd.read_csv("Desktop/jane-street-market-prediction/output_data.csv")


# In[15]:


pca = PCA(n_components = 1)
resp_pca = pca.fit_transform(resp)
y = (resp_pca > 0).astype("int")
y = y.reshape((-1,1))


# In[16]:


del resp
del pca
del resp_pca


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


# In[302]:


print("Train size:", y_train.shape[0], "; % trues:", np.sum(y_train)/y_train.shape[0])
print("Test size:", y_test.shape[0], "; % trues:", np.sum(y_test)/y_test.shape[0])


# In[77]:


y_values = pd.DataFrame(y_train,columns = ['y'])

x_values = pd.DataFrame(x_train)

combined_values = pd.concat([y_values,x_values],axis=1)


# In[78]:


combined_values


# In[87]:


combined_values.columns = combined_values.columns.astype(str)


# In[72]:


regr = linear_model.LinearRegression()
regr.fit(x_values, y_values)
print(regr.coef_)


# In[231]:


#Linear Regression

plt.scatter(combined_values['y'], combined_values['130'], color='red')
plt.grid(True)
plt.show()


# In[ ]:


#for x = 4 --> if x is greater than 18ish, probaly y=1. if x is less than -18ish, probably y = 0. 
#for x - 12 --> if x is greater than 32, y=1
#for x - 18 --> if x is greater than 38, y = 1. if x is less than -28, y = 0
#for x - 28 --> if x is greater than 22, y=1. if less than -22, y=0
#for x - 32 --> if x is greater than 61, y=0
#for x - 33 --> if x is greater than 60, y=0. 
#for x - 34 --> if x is greater than 30, y=0. if less than -30, y=1
#for x - 35 --> if x is greater than 60, y=0
#for x - 38 --> if x is greater than 20, y=1. if less than -19, y=0
#for x - 61 --> if x is greater than 900, y=0
#for x - 128 --> high x (over 40) leads to y=0
#for x - 126 --> high x (over 45) leads to y=0


# In[232]:


X = x_values.values.reshape(-1, 131)
y = y_values.values

ols = linear_model.LinearRegression()
model = ols.fit(X, y)


# In[233]:


print('Features                :  %s' % features)
print('Regression Coefficients : ', [x_values.shape for item in model.coef_])
print('R-squared               :  %.2f' % model.score(X, y))
print('Y-intercept             :  %.2f' % model.intercept_)
print('')


# In[234]:


#correlations

correlations = combined_values.corr()


# In[275]:


#random forest
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(x_values, y_values)

imp = rfpimp.importances(rf, x_values, y_values)


# In[235]:


#heatmap for all x values

fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(combined_values.corr(), center=0)
#ax.set_title(‘Multi-Collinearity of Car Attributes’)


# In[277]:


#pair plot

sns.pairplot(combined_values, hue = 'y')


# In[271]:


#Linear Regression

X = x_train[['feature_17','feature_27','feature_28','feature_33','feature_31']] #1777777,2777777,28,31,33
y = y_train['resp_4']
regr = linear_model.LinearRegression()
regr.fit(x_train, y)
print(regr.score(x_train,y))
print(regr.coef_)
plt.scatter(x_train['feature_27'],y_train['resp_4'])


# In[77]:


numeric = df.iloc[: , [0,1,5,9]]
sns.pairplot(numeric.dropna(), hue='Survived')


# In[253]:


len(x_values)


# In[255]:


clf = skl_lm.LogisticRegression(solver='newton-cg') ## Command for Logistic Regression. We are using Newton's method for computing MLE. You are not required to know what it is exactly for the purpose of this course.
clf.fit(x_values,y_train.ravel())
#prob = clf.predict_proba(x_test)


# In[262]:


fig = plt.figure(figsize=(12,5))
gs = mpl.gridspec.GridSpec(1, 4)
ax1 = plt.subplot(gs[0,:-2])
ax2 = plt.subplot(gs[0,-2])
ax3 = plt.subplot(gs[0,-1])


# In[271]:


#Logistic Regressions

#clf = skl_lm.LogisticRegression(solver='newton-cg') ## Command for Logistic Regression. We are using Newton's method for computing MLE. You are not required to know what it is exactly for the purpose of this course.
#clf.fit(x_values,y_train)
prob = clf.predict_proba(x_test)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
# Left plot
sns.regplot(x_values, y_values, order=1, ci=None,
            scatter_kws={'color':'orange'},
            line_kws={'color':'lightblue', 'lw':2}, ax=ax1)
# Right plot
ax2.scatter(x_values[17], y, color='orange')
ax2.plot(x_test, prob[:,1], color='lightblue')

for ax in fig.axes:
    ax.hlines(1, xmin=ax.xaxis.get_data_interval()[0],
              xmax=ax.xaxis.get_data_interval()[1], linestyles='dashed', lw=1)
    ax.hlines(0, xmin=ax.xaxis.get_data_interval()[0],
              xmax=ax.xaxis.get_data_interval()[1], linestyles='dashed', lw=1)
    ax.set_ylabel('Probability of default')
    ax.set_xlabel('Balance')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.])
    ax.set_xlim(xmin=-100)


# In[276]:


#LDA
X = x_values[1:130].to_numpy()
y = y_values.y.to_numpy()

lda = LinearDiscriminantAnalysis(solver='svd') ## SVD is another way of computing the LDA classification rule.
y_pred = lda.fit(X, y).predict(X)


# In[ ]:


#Visualize Decision Boundary

# Our 2-dimensional distribution will be over variables X and Y

x, y = np.meshgrid(x_values, y_train)

#Initialize seaborn facetplot
g = sns.FacetGrid(df, hue="y", size=10000, palette = 'colorblind') .map(plt.scatter,"resp_1", "resp_2","resp_3","resp_4", )  .add_legend()
my_ax = g.ax #Retrieving the faceplot axes

#Computing the predicted class function for each value on the grid
zz = np.array(  [predict_LDA_class( np.array([xx,yy]).reshape(-1,1), mu_list, Sigma, pi_list) 
                     for xx, yy in zip(np.ravel(X), np.ravel(Y)) ] )
    
#Reshaping the predicted class into the meshgrid shape
Z = zz.reshape(X.shape)


#Plot the filled and boundary contours
my_ax.contourf( x, resp, Z, 2, alpha = .1, colors = ('blue','green','red'))
my_ax.contour( x, resp, Z, 2, alpha = 1, colors = ('blue','green','red'))

# Addd axis and title
my_ax.set_xlabel('X')
my_ax.set_ylabel('Y')
my_ax.set_title('LDA and boundaries')

plt.show()


# In[ ]:




