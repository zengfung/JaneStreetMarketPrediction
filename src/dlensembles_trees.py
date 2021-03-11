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
from utils import fit_ensemble
from utils import ensemble_predict

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
## decision tree ensemble
print("\nDECISION TREE ENSEMBLE:")
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
params = {"criterion" : ["gini", "entropy"],
          "splitter" : ["best", "random"],
          "min_samples_split" : [2, 20, 200]}
ensemble_model = fit_ensemble(members, ensembleX_train, y_train, train_size, model, params)
# grid search results
ensemble_results = pd.DataFrame.from_dict(ensemble_model.cv_results_)
print(ensemble_results)
ensemble_results.to_csv("../results/ensembles_decisiontree.csv")
print("Best model score:", ensemble_model.best_score_)
print(ensemble_model.best_params_)
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
model = RandomForestClassifier()
params = {"n_estimators" : [50, 100, 500],
          "criterion" : ["gini", "entropy"],
          "min_samples_split" : [2, 20, 200],
          "max_features" : ["auto", "sqrt", "log2", None]}
ensemble_model = fit_ensemble(members, ensembleX_train, y_train, train_size, model, params)
# grid search results
ensemble_results = pd.DataFrame.from_dict(ensemble_model.cv_results_)
print(ensemble_results)
ensemble_results.to_csv("../results/ensembles_randomforest.csv")
print("Best model score:", ensemble_model.best_score_)
print(ensemble_model.best_params_)
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
model = AdaBoostClassifier()
params = {"n_estimators" : [50, 100, 500],
          "learning_rate" : [0.1, 1, 10], 
          "max_features" : ["auto", "sqrt", "log2", None]}
ensemble_model = fit_ensemble(members, ensembleX_train, y_train, train_size, model, params)
# grid search results
ensemble_results = pd.DataFrame.from_dict(ensemble_model.cv_results_)
print(ensemble_results)
ensemble_results.to_csv("../results/ensembles_adaboost.csv")
print("Best model score:", ensemble_model.best_score_)
print(ensemble_model.best_params_)
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
model = GradientBoostingClassifier()
params = {"loss" : ["deviance", "exponential"],
          "learning_rate" : [0.01, 1, 10], 
          "n_estimators" : [50, 100, 500],
          "subsample" : [0.5, 0.75, 1.0]}
ensemble_model = fit_ensemble(members, ensembleX_train, y_train, train_size, model, params)
# grid search results
ensemble_results = pd.DataFrame.from_dict(ensemble_model.cv_results_)
print(ensemble_results)
ensemble_results.to_csv("../results/ensembles_gradientboosting.csv")
print("Best model score:", ensemble_model.best_score_)
print(ensemble_model.best_params_)
# predict output using ensemble model
yhat_train = ensemble_predict(members, ensemble_model, ensembleX_train, train_size)
train_acc = accuracy_score(y_train, yhat_train)
print("Train accuracy:", train_acc)
yhat_test = ensemble_predict(members, ensemble_model, ensembleX_test, test_size)
test_acc = accuracy_score(y_test, yhat_test)
print("Test accuracy:", test_acc)
results["GradientBoosting(E)"] = {"Train": train_acc, "Test": test_acc}

# results to data frame
df = pd.DataFrame.from_dict(results, orient = "index")
df = df.reset_index()
df = df.rename(columns = {"index": "Model", "Train": "Train Accuracy", "Test": "Test Accuracy"})
# save results
df.to_csv("../results/ensembleresults.csv", mode = "a", header = False)