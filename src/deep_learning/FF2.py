# deep ANN model
# remove warnings from tensorflow and sklearn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd

x = pd.read_csv("../../dataset/input_data.csv").to_numpy()
resp = pd.read_csv("../../dataset/output_data.csv")

##
from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components = 1)
resp_pca = pca.fit_transform(resp)
y = (resp_pca > 0).astype("int")
y = y.reshape((-1,1))

del resp
del pca
del resp_pca

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


print("Train size:", y_train.shape[0], "; % trues:", np.sum(y_train)/y_train.shape[0])
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

def fit_model(x_train, y_train, epochs = 100, batch_size = 1024):
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
        metrics = [tf.keras.metrics.AUC(name="AUC"), "accuracy"]
        )
    history = model.fit(
        x = x_train, 
        y = y_train, 
        epochs = epochs, 
        batch_size = batch_size,
        validation_split = 0.2,
        callbacks = [EarlyStopping('accuracy', patience=10, restore_best_weights = True)],
        verbose = 2
        )
    
    return model, history

model, history = fit_model(x_train, y_train, epochs = 500, batch_size = 1024)

##
model.summary()
model.save("../../models/FF2.h5")

#%%
from sklearn.metrics import accuracy_score

yhat_train = model.predict(x_train)
yhat_train = (yhat_train > 0.5).astype("int")
acc = accuracy_score(y_train, yhat_train)
print("Train accuracy score:", acc)

yhat_test = model.predict(x_test)
yhat_test = (yhat_test > 0.5).astype("int")
acc = accuracy_score(y_test, yhat_test)
print("Test accuracy score:", acc)

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

fig.savefig("../../results/FF2.png")
fig.show()