# remove warnings from tensorflow and sklearn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from utils import load_keras_models
from utils import transform_inputs
from utils import individual_prediction

x = pd.read_csv("../dataset/input_data.csv").to_numpy()
resp = pd.read_csv("../dataset/output_data.csv")
# run PCA on resp values + set action = 1 if PCA'd resp value > 0
pca = PCA(n_components = 1)
resp_pca = pca.fit_transform(resp)
y = (resp_pca > 0).astype("int")
y = y.reshape((-1,1))
# delete unneeded variables
del resp
del pca
del resp_pca
# split data into train, valid, test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
train_size = y_train.shape[0]
test_size = y_test.shape[0]
print("Train size:", train_size, "; % trues:", np.sum(y_train)/train_size)
print("Test size:", test_size, "; % trues:", np.sum(y_test)/test_size)

# majority voting
def majority_votes(members, x, n):
	k = len(members)		# number of models
	result = np.zeros((n,1))
	for (i, model) in enumerate(members):
		yhat = model.predict(x[i])
		yhat = (yhat > 0.5).astype("int")
		result += yhat
	result = (result > k/2).astype("int")
	return result

# average voting
def average_votes(members, x, n):
	k = len(members)		# number of models
	result = np.zeros((n, 1))
	for (i, model) in enumerate(members):
		yhat = model.predict(x[i])
		result += yhat
	result = result / k 	# mean scores
	result = (result > 0.5).astype("int")
	return result

# models to be loaded
model_files = ["../models/dlmodel2.h5",
			   "../models/dlmodel3.h5",
			   "../models/dlmodel6.h5",
			   "../models/dlmodel8.h5",
			   "../models/dlmodel10.h5"]
model_types = ["split", "combine", "pca_split", "pca_split", "pca_split"]
# load model and transform inputs
members = load_keras_models(model_files)
[ensembleX_train, ensembleX_test] = transform_inputs(x_train, x_test, model_types)

# result dictionary to keep track of model results
results = {}
# predict output using indivdual models
print("\nINDIVIDUAL MODEL RESULTS:")
individual_prediction(members, ensembleX_train, y_train, ensembleX_test, y_test, results)

# voting results
print("\nMODEL VOTING RESULTS:")
# predict output using majority votes
print("\nMAJORITY VOTING RESULTS:")
yhat_train = majority_votes(members, ensembleX_train, train_size)
train_acc = accuracy_score(y_train, yhat_train)
print("Train accuracy:", train_acc)
yhat_test = majority_votes(members, ensembleX_test, test_size)
test_acc = accuracy_score(y_test, yhat_test)
print("Test accuracy:", test_acc)
results["Majority(V)"] = {"Train": train_acc, "Test": test_acc}

# predict output using average votes
print("\nAVERAGE VOTING RESULTS:")
yhat_train = average_votes(members, ensembleX_train, train_size)
train_acc = accuracy_score(y_train, yhat_train)
print("Train accuracy:", train_acc)
yhat_test = average_votes(members, ensembleX_test, test_size)
test_acc = accuracy_score(y_test, yhat_test)
print("Test accuracy:", test_acc)
results["Average(V)"] = {"Train": train_acc, "Test": test_acc}

# results to data frame
df = pd.DataFrame.from_dict(results, orient = "index")
df = df.reset_index()
df = df.rename(columns = {"index": "Model", "Train": "Train Accuracy", "Test": "Test Accuracy"})
# save results
df.to_csv("../results/ensembleresults.csv")