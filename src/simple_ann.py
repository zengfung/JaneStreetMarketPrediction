# simple ANN model
import pandas as pd

x = pd.read_csv("../dataset/input_data.csv").to_numpy()
resp = pd.read_csv("../dataset/output_data.csv")

##
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import OneHotEncoder

pca = PCA(n_components = 1)
resp_pca = pca.fit_transform(resp)
y = (resp_pca > 0).astype("int")
y = y.reshape((-1,1))
onehot_encoder = OneHotEncoder(sparse=False)
y = onehot_encoder.fit_transform(y)

del resp
del pca
del resp_pca

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.3, random_state = 2)

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
        Dense(units = 2, activation = "softmax")
        ])
    
    opt = Adam(learning_rate = 0.01)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])
    model.fit(x_train, y_train, epochs = 200, batch_size = 1024,
              validation_data = (x_valid, y_valid),
              callbacks = [EarlyStopping(monitor="val_accuracy", patience=15)])
    
    return model


model = simple_ann(x_train, y_train, x_valid, y_valid)

#%%
from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype("int")
acc = accuracy_score(y_test, y_pred)
print("Test accuracy score:", acc)

##
model.summary()
model.save("../models/simple_ann.h5")