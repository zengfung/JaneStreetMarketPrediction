# convolutional-regular neural network
import pandas as pd

x = pd.read_csv("../dataset/input_data.csv", nrows=10000).to_numpy()
resp = pd.read_csv("../dataset/output_data.csv", nrows= 10000)

#%%
# run PCA on resp values + set action = 1 if PCA'd resp value > 0
from sklearn.decomposition import PCA

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
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
[w_train, ts_train] = [x_train[:,0], x_train[:,1:]]
[w_test, ts_test] = [x_test[:,0], x_test[:,1:]]

print("Train size:", y_train.shape[0], "; % trues:", np.sum(y_train)/y_train.shape[0])
print("Test size:", y_test.shape[0], "; % trues:", np.sum(y_test)/y_test.shape[0])

#%%
# building model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def fit_model(x_train, y_train, epochs = 100, batch_size = 1024):
    # convolutional layers for time-series
    ts = Sequential([
        Input(shape = (130,1)),
        # 1st set of convolution layer (16 -> 16 -> Maxpool)
        Conv1D(filters = 16, kernel_size = 2, strides = 1, padding = "valid",
               kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        Conv1D(filters = 16, kernel_size = 2, strides = 1, padding = "valid",
                kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        MaxPooling1D(pool_size = 2, strides = 1, padding = "valid"),
        # 2nd set of convolutional layer (64 -> 64 -> Maxpool)
        Conv1D(filters = 64, kernel_size = 3, strides = 1, padding = "valid",
               kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        Conv1D(filters = 64, kernel_size = 3, strides = 1, padding = "valid",
                kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        MaxPooling1D(pool_size = 2, strides = 1, padding = "valid"),
        Flatten()
        ])
    
    # regular layer for weight
    w = Sequential([
        Input(shape = (1,)),
        Dense(units = 4, kernel_initializer = GlorotNormal())
        ])
    
    # concatenate ts and w
    model_concat = Concatenate(axis = -1)([ts.output, w.output])
    # 1st set of layers (128 -> 128 -> 128 -> 128 -> Dropout)
    model_concat = Dense(units = 128, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dense(units = 128, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dense(units = 128, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dense(units = 128, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dropout(0.2)(model_concat)
    # 2nd set of layers (64 -> 64 -> 64 -> 64 -> Dropout)
    model_concat = Dense(units = 64, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dense(units = 64, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dense(units = 64, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dense(units = 64, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dropout(0.2)(model_concat)
    # 3rd set of layers (32 -> 32 -> 32 -> 32 -> Dropout)
    model_concat = Dense(units = 32, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dense(units = 32, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dense(units = 32, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dense(units = 32, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dropout(0.2)(model_concat)
    # output layer
    model_concat = Dense(units = 1, activation = "sigmoid")(model_concat)
    
    # define full model
    model = Model(inputs = [ts.input, w.input], outputs = model_concat)
    
    # fit model
    opt = Adam(learning_rate = 0.01)
    model.compile(loss = "binary_crossentropy", optimizer = opt, 
                  metrics = [tf.keras.metrics.AUC(name="AUC"), "accuracy"])
    model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size,
              validation_split = 0.2,
              callbacks = [EarlyStopping('accuracy', patience=10, restore_best_weights = True)],
              verbose = 1)
    
    return model

model = fit_model([ts_train, w_train], y_train)

#%%
from sklearn.metrics import accuracy_score

yhat_train = model.predict([ts_train, w_train])
yhat_train = (yhat_train > 0.5).astype("int")
acc = accuracy_score(y_train, yhat_train)
print("Train accuracy score:", acc)

yhat_test = model.predict([ts_test, w_test])
yhat_test = (yhat_test > 0.5).astype("int")
acc = accuracy_score(y_test, yhat_test)
print("Test accuracy score:", acc)

##
model.summary()
model.save("../models/deephybrid.h5")