# RBF SVM model
import pandas as pd

x = pd.read_csv("../dataset/input_data.csv").to_numpy()
resp = pd.read_csv("../dataset/output_data.csv").to_numpy()

##
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components = 1)
resp_pca = pca.fit_transform(resp)
y = (resp_pca > 0).astype("int")
y = np.ravel(y)

##
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.95, random_state = 1)
print("Train size:", y_train.shape[0], "; % trues:", np.sum(y_train)/y_train.shape[0])
print("Test size:", y_test.shape[0], "; % trues:", np.sum(y_test)/y_test.shape[0])

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# fit model
clf = SVC(kernel = "rbf")
clf.fit(x_train, y_train)

# predictions on training set
yhat_train = clf.predict(x_train)
print("Train MSE:", accuracy_score(y_train, yhat_train))

# predictions on test set
yhat_test = clf.predict(x_test)
print("Test MSE:", accuracy_score(y_test, yhat_test))