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
model_files = ["../models/dlmodel10_1.h5",
			   "../models/dlmodel10_2.h5",
			   "../models/dlmodel10_3.h5",
			   "../models/dlmodel10_4.h5",
			   "../models/dlmodel10_5.h5"]
model_types = ["pca_split", "pca_split", "pca_split", "pca_split", "pca_split"]
# load model and transform inputs
members = load_keras_models(model_files)
[ensembleX_train, ensembleX_test] = transform_inputs(x_train, x_test, model_types)

# result dictionary to keep track of model results
results = {}
# LDA ensemble
print("\nLDA ENSEMBLE:")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
ensemble_model = fit_ensemble(members, ensembleX_train, y_train, train_size, LinearDiscriminantAnalysis(), {})
# predict output using ensemble model
yhat_train = ensemble_predict(members, ensemble_model, ensembleX_train, train_size)
train_acc = accuracy_score(y_train, yhat_train)
print("Train accuracy:", train_acc)
yhat_test = ensemble_predict(members, ensemble_model, ensembleX_test, test_size)
test_acc = accuracy_score(y_test, yhat_test)
print("Test accuracy:", test_acc)
results["LDA(E)"] = {"Train": train_acc, "Test": test_acc}

# results to data frame
df = pd.DataFrame.from_dict(results, orient = "index")
df = df.reset_index()
df = df.rename(columns = {"index": "Model", "Train": "Train Accuracy", "Test": "Test Accuracy"})
# save results
df.to_csv("../results/ensembleresults.csv", mode = "a", header = False)