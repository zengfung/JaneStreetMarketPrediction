# AlexNet-inspired convolutional-regular neural network
import pandas as pd

x = pd.read_csv("../dataset/input_data.csv", nrows = 10000).to_numpy()
resp = pd.read_csv("../dataset/output_data.csv", nrows = 10000)

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
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

def fit_model(x_train, y_train, epochs = 100, batch_size = 1024):
    # convolutional layers for time-series
    ts = Sequential([
        Input(shape = (130,1)),
        # 1st set of convolution layer (96 -> MaxPool -> Dropout)
        Conv1D(filters = 96, kernel_size = 6, strides = 4, padding = "valid",
               kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        MaxPooling1D(pool_size = 2, strides = 2, padding = "valid"),
        SpatialDropout1D(0.2),
        
        # 2nd set of convolutional layer (256 -> MaxPool)
        Conv1D(filters = 256, kernel_size = 5, strides = 1, padding = "same",
               kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        MaxPooling1D(pool_size = 2, strides = 2, padding = "valid"),
        SpatialDropout1D(0.2),

        # 3rd set of convolutional layer (384 -> 384 -> 256 -> MaxPool)
        Conv1D(filters = 384, kernel_size = 3, strides = 1, padding = "same",
               kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        Conv1D(filters = 384, kernel_size = 3, strides = 1, padding = "same",
               kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        SpatialDropout1D(0.2),
        Conv1D(filters = 256, kernel_size = 3, strides = 1, padding = "same",
               kernel_initializer = GlorotNormal()),
        Activation(tf.keras.activations.relu),
        MaxPooling1D(pool_size = 3, strides = 2, padding = "valid"),
        
        # Flatten layer
        Flatten()
        ])
    
    # regular layer for weight
    w = Sequential([
        Input(shape = (1,)),
        Dense(units = 4, kernel_initializer = GlorotNormal())
        ])
    
    # concatenate ts and w
    model_concat = Concatenate(axis = -1)([ts.output, w.output])
    # 1st set of layers (256 -> Dropout -> 256 -> Dropout )
    model_concat = Dense(units = 256, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dense(units = 256, kernel_initializer = GlorotNormal())(model_concat)
    model_concat = BatchNormalization()(model_concat)
    model_concat = Activation(tf.keras.activations.relu)(model_concat)
    model_concat = Dropout(0.2)(model_concat)
    # output layer
    model_concat = Dense(units = 1, activation = "sigmoid")(model_concat)
    
    # define full model
    model = Model(inputs = [ts.input, w.input], outputs = model_concat)
    
    # fit model
    opt = SGD(learning_rate = 0.05, momentum = 0.9, decay = 0.0005)
    model.compile(loss = "binary_crossentropy", optimizer = opt, 
                  metrics = [tf.keras.metrics.AUC(name="AUC"), "accuracy"])
    history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size,
              validation_split = 0.2,
              callbacks = [EarlyStopping('accuracy', patience=10, restore_best_weights = True)],
              verbose = 2)
    
    return model, history

model, history = fit_model([ts_train, w_train], y_train, 100, 1024)

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
model.save("../models/dlmodel7.h5")

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

fig.savefig("../results/dlmodel7.png")
fig.show()