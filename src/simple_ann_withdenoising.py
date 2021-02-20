# simple ANN model
import pandas as pd

x = pd.read_csv("../dataset/input_data.csv", nrows=2000).to_numpy()
resp = pd.read_csv("../dataset/output_data.csv", nrows= 2000)

#%%
# run PCA on resp values + set action = 1 if PCA'd resp value > 0
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components = 1)
resp_pca = pca.fit_transform(resp)
y = (resp_pca > 0).astype("int")
y = y.reshape((-1,1))

del resp
del pca
del resp_pca

#%%
# split data into train, valid, test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = 2)

#%%
# denoise signals (refer to utils.py for more info)
from utils import denoise_test_signals, denoise_signals
x_train[:,1:], x_valid[:,1:], t = denoise_signals(x_train[:,1:], x_valid[:,1:], wt = "haar")
x_test[:,1:] = denoise_test_signals(x_test[:,1:], wt = "db1", t = t)

print("Train size:", y_train.shape[0], "; % trues:", np.sum(y_train)/y_train.shape[0])
print("Valid size:", y_valid.shape[0], "; % trues:", np.sum(y_valid)/y_valid.shape[0])
print("Test size:", y_test.shape[0], "; % trues:", np.sum(y_test)/y_test.shape[0])
#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def simple_ann(x_train, y_train, x_valid, y_valid):
    model = Sequential([
        Input(shape=(131,)),
        Dense(units = 64, kernel_initializer = GlorotNormal()),
        BatchNormalization(),
        Activation(tf.keras.activations.relu),
        Dense(units = 32, kernel_initializer = GlorotNormal()),
        BatchNormalization(),
        Activation(tf.keras.activations.relu),
        Dense(units = 16, kernel_initializer = GlorotNormal()),
        BatchNormalization(),
        Activation(tf.keras.activations.relu),
        Dense(units = 1, activation = "sigmoid")
        ])
    
    opt = Adam(learning_rate = 0.01)
    model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])
    model.fit(x_train, y_train, epochs = 100, batch_size = 1024,
              validation_data = (x_valid, y_valid),
              verbose = 2)
    
    return model


model = simple_ann(x_train, y_train, x_valid, y_valid)

#%%
from sklearn.metrics import accuracy_score

yhat_train = model.predict(x_train)
yhat_train = (yhat_train > 0.5).astype("int")
acc = accuracy_score(y_train, yhat_train)
print("Train accuracy score:", acc)

yhat_valid = model.predict(x_valid)
yhat_valid = (yhat_valid > 0.5).astype("int")
acc = accuracy_score(y_valid, yhat_valid)
print("Valid accuracy score:", acc)

yhat_test = model.predict(x_test)
yhat_test = (yhat_test > 0.5).astype("int")
acc = accuracy_score(y_test, yhat_test)
print("Test accuracy score:", acc)

##
model.summary()
model.save("../models/simple_ann_withdenoising.h5")