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

# obtain outputs from deep learning models
def get_ensemble_input(members, x, n):
	k = len(members)
	result = np.zeros((n,k))
	for (i, model) in enumerate(members):
		yhat = model.predict(x[i])
		result[:,i] = yhat.reshape((-1,))
	return result

# fit ensemble model
def fit_ensemble(members, x, y, n, model_type):
	# get ensemble model input
	ensemble_input = get_ensemble_input(members, x, n)
	# fit ensemble model
	model_type.fit(ensemble_input, y)
	return model_type

# make prediction using ensemble model
def ensemble_predict(members, model, x, n):
	# get ensemble model input
	ensemble_input = get_ensemble_input(members, x, n)
	# make prediction
	yhat = model.predict(ensemble_input)
	return yhat

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

# fit ensemble models
print("\nSTACKING ENSEMBLE RESULTS:")

# logistic regression ensemble
print("\nLOGISTIC REGRESSION ENSEMBLE:")
from sklearn.linear_model import LogisticRegression
ensemble_model = fit_ensemble(members, ensembleX_train, y_train, train_size, LogisticRegression())
# predict output using ensemble model
yhat_train = ensemble_predict(members, ensemble_model, ensembleX_train, train_size)
train_acc = accuracy_score(y_train, yhat_train)
print("Train accuracy:", train_acc)
yhat_test = ensemble_predict(members, ensemble_model, ensembleX_test, test_size)
test_acc = accuracy_score(y_test, yhat_test)
print("Test accuracy:", test_acc)
results["LogisticRegression(E)"] = {"Train": train_acc, "Test": test_acc}

# decision tree ensemble
print("\nDECISION TREE ENSEMBLE:")
from sklearn.tree import DecisionTreeClassifier
ensemble_model = fit_ensemble(members, ensembleX_train, y_train, train_size, DecisionTreeClassifier())
# predict output using ensemble model
yhat_train = ensemble_predict(members, ensemble_model, ensembleX_train, train_size)
train_acc = accuracy_score(y_train, yhat_train)
print("Train accuracy:", train_acc)
yhat_test = ensemble_predict(members, ensemble_model, ensembleX_test, test_size)
test_acc = accuracy_score(y_test, yhat_test)
print("Test accuracy:", test_acc)
results["DecisionTree(E)"] = {"Train": train_acc, "Test": test_acc}

# random forest ensemble
print("\nRANDOM FOREST ENSEMBLE:")
from sklearn.ensemble import RandomForestClassifier
ensemble_model = fit_ensemble(members, ensembleX_train, y_train, train_size, RandomForestClassifier())
# predict output using ensemble model
yhat_train = ensemble_predict(members, ensemble_model, ensembleX_train, train_size)
train_acc = accuracy_score(y_train, yhat_train)
print("Train accuracy:", train_acc)
yhat_test = ensemble_predict(members, ensemble_model, ensembleX_test, test_size)
test_acc = accuracy_score(y_test, yhat_test)
print("Test accuracy:", test_acc)
results["RandomForest(E)"] = {"Train": train_acc, "Test": test_acc}

# adaboost ensemble
print("\nADABOOST ENSEMBLE:")
from sklearn.ensemble import AdaBoostClassifier
ensemble_model = fit_ensemble(members, ensembleX_train, y_train, train_size, AdaBoostClassifier())
# predict output using ensemble model
yhat_train = ensemble_predict(members, ensemble_model, ensembleX_train, train_size)
train_acc = accuracy_score(y_train, yhat_train)
print("Train accuracy:", train_acc)
yhat_test = ensemble_predict(members, ensemble_model, ensembleX_test, test_size)
test_acc = accuracy_score(y_test, yhat_test)
print("Test accuracy:", test_acc)
results["AdaBoost(E)"] = {"Train": train_acc, "Test": test_acc}

# gradient boosting ensemble
print("\nGRADIENT BOOSTING ENSEMBLE:")
from sklearn.ensemble import GradientBoostingClassifier
ensemble_model = fit_ensemble(members, ensembleX_train, y_train, train_size, GradientBoostingClassifier())
# predict output using ensemble model
yhat_train = ensemble_predict(members, ensemble_model, ensembleX_train, train_size)
train_acc = accuracy_score(y_train, yhat_train)
print("Train accuracy:", train_acc)
yhat_test = ensemble_predict(members, ensemble_model, ensembleX_test, test_size)
test_acc = accuracy_score(y_test, yhat_test)
print("Test accuracy:", test_acc)
results["GradientBoosting(E)"] = {"Train": train_acc, "Test": test_acc}

# SVM ensemble
print("\nSVM ENSEMBLE:")
from sklearn.svm import SVC
ensemble_model = fit_ensemble(members, ensembleX_train, y_train, train_size, SVC(kernel = "rbf"))
# predict output using ensemble model
yhat_train = ensemble_predict(members, ensemble_model, ensembleX_train, train_size)
train_acc = accuracy_score(y_train, yhat_train)
print("Train accuracy:", train_acc)
yhat_test = ensemble_predict(members, ensemble_model, ensembleX_test, test_size)
test_acc = accuracy_score(y_test, yhat_test)
print("Test accuracy:", test_acc)
results["SVM-RBF(E)"] = {"Train": train_acc, "Test": test_acc}

#%%
# plot results
import matplotlib.pyplot as plt

df = pd.DataFrame.from_dict(results, orient = "index")
df = df.reset_index()
df = df.rename(columns = {"index": "Model", "Train": "Train Accuracy", "Test": "Test Accuracy"})

ind = np.arange(len(df))
width = 0.4

model = np.asarray(df["Model"])
train = np.asarray(df["Train Accuracy"])
test = np.asarray(df["Test Accuracy"])

fig, ax = plt.subplots(figsize = (20,10))
ax.bar(ind - width, train, width, color = "yellow", edgecolor = "black")
ax.bar(ind, test, width, color = "red", edgecolor = "black")
ax.legend(["Train Accuracy", "Test Accuracy"], fontsize = 14)
ax.set_ylim(0,1.1)
ax.set_ylabel("Accuracy", fontsize = 14)
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize = 14)
ax.set_xticks(ind)
ax.set_xlabel("Models", fontsize = 14)
ax.set_xticklabels(model, rotation = 20, fontsize = 14, ha = "right")
ax.set_title("Comparison of Accuracies between DL models and their Ensembles", fontsize = 18)

for i in range(len(df)):
	ax.text(ind[i] - width, train[i] + 0.01, str(round(train[i], 3)),
		ha = "center", va = "bottom", size = 14, rotation = "vertical")
	ax.text(ind[i], test[i] + 0.01, str(round(test[i], 3)),
		ha = "center", va = "bottom", size = 14, rotation = "vertical")

# save results
df.to_csv("../results/ensembleresults.csv")
fig.savefig("../results/ensemblechart.png")
fig.show()