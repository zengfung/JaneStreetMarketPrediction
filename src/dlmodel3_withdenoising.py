# deep ANN model
import pandas as pd

x = pd.read_csv("../dataset/input_data.csv", nrows = 10000).to_numpy()
resp = pd.read_csv("../dataset/output_data.csv", nrows = 10000)

# pca
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components = 1)
resp_pca = pca.fit_transform(resp)
y = (resp_pca > 0).astype("int")
y = y.reshape((-1,1))

del resp
del pca
del resp_pca

# split data to train, valid, test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = 2)

# denoise signals (refer to utils.py for more info)
from utils import denoise_test_signals, denoise_signals
x_train[:,1:], x_valid[:,1:], t = denoise_signals(x_train[:,1:], x_valid[:,1:], wt = "sym2")
x_test[:,1:] = denoise_test_signals(x_test[:,1:], wt = "db1", t = t)

print("Train size:", y_train.shape[0], "; % trues:", np.sum(y_train)/y_train.shape[0])
print("Valid size:", y_valid.shape[0], "; % trues:", np.sum(y_valid)/y_valid.shape[0])
print("Test size:", y_test.shape[0], "; % trues:", np.sum(y_test)/y_test.shape[0])
#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def fit_model(x_train, y_train, x_valid, y_valid, epochs = 200, batch_size = 1024):
    model = Sequential([
        Input(shape=(131,)),
        # hidden layer 1
        Dense(units = 128, kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        BatchNormalization(),
        # hidden layer 2
        Dense(units = 128, kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        BatchNormalization(),
        # hidden layer 3
        Dense(units = 128, kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        BatchNormalization(),
        # hidden layer 4
        Dense(units = 128, kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        BatchNormalization(),
        # dropout
        Dropout(0.2),
        
        # hidden layer 5
        Dense(units = 64, kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        BatchNormalization(),
        # hidden layer 6
        Dense(units = 64, kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        BatchNormalization(),      
        # hidden layer 7
        Dense(units = 64, kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        BatchNormalization(),
        # hidden layer 8
        Dense(units = 64, kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        BatchNormalization(),
        Dropout(0.2),
        
        # hidden layer 9
        Dense(units = 32, kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        BatchNormalization(),
        # hidden layer 10
        Dense(units = 32, kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        BatchNormalization(),      
        # hidden layer 11
        Dense(units = 32, kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        BatchNormalization(),
        # hidden layer 12
        Dense(units = 32, kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        BatchNormalization(),
        Dropout(0.2),
        
        # output
        Dense(units = 1, activation = "sigmoid")
        ])
    
    opt = Adam(learning_rate = 0.01)
    model.compile(
        loss = "binary_crossentropy", 
        optimizer = opt,
        metrics = [tf.keras.metrics.AUC(name="AUC"), "accuracy"])
    model.fit(
        x = x_train, 
        y = y_train, 
        epochs = epochs, 
        batch_size = batch_size,
        validation_data = (x_valid, y_valid),
        callbacks = [EarlyStopping('accuracy', patience=10, restore_best_weights = True)],
        verbose = 2)
    
    return model

model = fit_model(x_train, y_train, x_valid, y_valid, epochs = 1000, batch_size = 1024)

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
# model.summary()
model.save("../models/dlmodel3_withdenoising.h5")