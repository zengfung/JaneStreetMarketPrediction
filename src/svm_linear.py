# linear SVM model
import pandas as pd

x = pd.read_csv("../dataset/input_data.csv")
resp = pd.read_csv("../dataset/output_data.csv")

##
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components = 1)
resp_pca = pca.fit_transform(resp)
y = (resp_pca > 0).astype("int")
y = np.ravel(y)

##
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.995, random_state = 1)
print("Number of training data:", x_train.shape[0])
print("Trues proportion in training data:", sum(y_train)/len(y_train))
print("Trues propotion in test data:", sum(y_test)/len(y_test))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# fit model
clf = SVC(kernel = "linear")
clf.fit(x_train, y_train)

# predictions on training set
yhat_train = clf.predict(x_train)
print("Train MSE:", accuracy_score(y_train, yhat_train))

# predictions on test set
yhat_test = clf.predict(x_test)
print("Test MSE:", accuracy_score(y_test, yhat_test))