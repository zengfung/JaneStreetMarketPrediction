# stacking ensemble of DL models

## import data
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

x = pd.read_csv("../dataset/input_data.csv", nrows = 1000).to_numpy()
resp = pd.read_csv("../dataset/output_data.csv", nrows = 1000)
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
print("Train size:", y_train.shape[0], "; % trues:", np.sum(y_train)/y_train.shape[0])
print("Test size:", y_test.shape[0], "; % trues:", np.sum(y_test)/y_test.shape[0])

#%%
# load models
def load_keras_models(names):
    all_models = list()
    for filename in names:
        model = load_model(filename)
        all_models.append(model)
        print("Loaded {0}".format(filename))
    return all_models

# transforming inputs
def transform_inputs(train, test, types):
    train_stack = list()
    test_stack = list()
    for input_type in types:
        if input_type == "combine":
            train_stack.extend([train])
            test_stack.extend([test])
            print("combine-type transform success!")
        elif input_type == "split":
            [w_train, ts_train] = [train[:,0], train[:,1:]]
            [w_test, ts_test] = [test[:,0], test[:,1:]]
            train_stack.extend([ts_train, w_train])
            test_stack.extend([ts_test, w_test])
            print("split-type transform success!")
        elif input_type == "pca_split":
        	[w_train, ts_train] = [train[:,0], train[:,1:]]
        	[w_test, ts_test] = [test[:,0], test[:,1:]]
        	pca = PCA(n_components = 57).fit(ts_train)
        	tsf_train = pca.transform(ts_train)
        	tsf_test = pca.transform(ts_test)
        	train_stack.extend([tsf_train, w_train])
        	test_stack.extend([tsf_test, w_test])
        	print("pca_split-type transform success!")
        else: 
        	raise Exception("Invalid transform type!")
    return [train_stack, test_stack]

# define the stacked model
def define_stacked_model(members):
	# update all layers in all models to not be trainable
	for i in range(len(members)):
		model = members[i]
		for layer in model.layers:
			layer.trainable = False
			layer._name = "e" + str(i+1) + "_" + layer._name
	# define multi-headed input
	ensemble_visible = [model.input for model in members]
	# concatenate merge output from each model
	ensemble_outputs = [model.output for model in members]
	merge = Concatenate(axis = -1)(ensemble_outputs)
	hidden = Dense(16, activation = "relu")(merge)
	output = Dense(1, activation = "sigmoid")(hidden)
	# define the full model that binds everything
	model = Model(inputs = ensemble_visible, outputs = output)
	# compile model
	opt = SGD(learning_rate = 0.5, momentum = 0.9, decay = 0.0005, nesterov = True)
	model.compile(
        loss = "binary_crossentropy", 
        optimizer = opt,
        metrics = [tf.keras.metrics.AUC(name="AUC"), "accuracy"]
        )
	return model

# fit the stacked model
def fit_stacked_model(model, x, y, epochs = 100, batch_size = 1024):
	history = model.fit(
        x = x_train, 
        y = y_train, 
        epochs = epochs, 
        batch_size = batch_size,
        validation_split = 0.2,
        callbacks = [EarlyStopping('val_accuracy', patience=50, restore_best_weights = True)],
        verbose = 1
        )
	return history

# make predictions with stacked model
def predict_stacked_model(model, x, y):
	yhat = model.predict(x)
	yhat = (yhat > 0.5).astype("int")
	acc = accuracy_score(y, yhat)
	return acc

# models to be loaded
model_files = ["../models/dlmodel3.h5",
			   "../models/dlmodel2.h5",
			   "../models/dlmodel6.h5"]
model_types = ["combine", "split", "pca_split"]

# load model and transform inputs
members = load_keras_models(model_files)
[ensembleX_train, ensembleX_test] = transform_inputs(x_train, x_test, model_types)
# define and fit model
stacked_model = define_stacked_model(members)
stacked_model.summary()
# plot_model(stacked_model, show_shapes = True, to_file = "stackingensemble_graph.png")
history = fit_stacked_model(stacked_model, ensembleX_train, y_train)
stacked_model.save("../models/stackingensemble.h5")
# make predictions
acc_train = predict_stacked_model(stacked_model, ensembleX_train, y_train)
print("Stacked training accuracy:", acc_train)
acc_test = predict_stacked_model(stacked_model, ensembleX_test, y_test)
print("Stacked test accuracy:", acc_test)

#%%
# plot training vs validation accuracy
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle("Training vs Validation Data")

ax1.plot(history.history["accuracy"], "r-", label = "Train")
ax1.plot(history.history["val_accuracy"], "b-", label = "Val.")
ax1.set_xlabel("EPOCH", fontsize = 8)
ax1.set_ylabel("Accuracy", fontsize = 8)
ax1.set_title("Accuracy", fontsize = 10)
ax1.legend(fontsize = 8)

ax2.plot(history.history["loss"], "r-", label = "Train")
ax2.plot(history.history["val_loss"], "b-", label = "Val.")
ax2.set_xlabel("EPOCH", fontsize = 8)
ax2.set_ylabel("Loss", fontsize = 8)
ax2.set_title("Loss", fontsize = 10)
ax2.legend(fontsize = 8)

fig.savefig("../results/stackingensemble.png")
fig.show()