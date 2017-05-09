import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization as BatchNorm
from keras.optimizers import RMSprop, Adam
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

# load and set up data
shuffle = False  # whether to randomize the order of data
stack_data = 100  # stack training data with itself
data = pd.read_csv('VetoGA_147PtAu_Hull.csv')
data = data.sort('index', ascending=True)
N = data.shape[0]
if(shuffle):
    data = data.sample(N)

# PREPROCESSING

X = scale(np.array(data.iloc[:, :-2]))
Y = np.array(data.iloc[:, -2])
idxs = np.array(data.iloc[:, -1])

# build training and test sets
train_num = int(.75 * N)  # how much data for training
remove = int(0.0 * N)  # how much data to excerpt between train and test
test_num = N - train_num - remove
X_train = X[:train_num, :]
Y_train = Y[:train_num]
X_test = X[-test_num:, :]
Y_test = Y[-test_num:]

# here we just concatenate the training data with itself stack_data times
# so that our readout looks a bit nicer
if stack_data > 1:
    new_train = np.zeros((stack_data*train_num, X_train.shape[1]))
    new_y = np.zeros(stack_data*train_num)
    for i in range(stack_data):
        new_train[i*train_num:(i+1)*train_num, :] = X_train
        new_y[i*train_num:(i+1)*train_num] = Y_train
    X_train = new_train
    Y_train = new_y

# BUILD NEURAL NETWORK

# initialize the model
model = Sequential()

# add dense hidden layers
model.add(Dense(48, input_dim=X.shape[1]))
# model.add(Dropout(0.2))
model.add(BatchNorm())
model.add(Activation('relu'))

# model.add(Dense(32))
# model.add(Dropout(0.3))
# model.add(BatchNorm())
# model.add(Activation('relu'))

# add output layer
model.add(Dense(1))

# TRAIN NEURAL NETWORK

# define optimizer (can tune optimizer parameters, e.g. learning rate)
optimizer = Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='mse')

# Workaround for bug in current version of keras, needed with BatchNorm layers
keras.backend.get_session().run(tf.initialize_all_variables())

# train the model
model.fit(X_train, Y_train, nb_epoch=20, verbose=1, batch_size=1024)

# EVALUATE

# predictions on test set
yhat = model.predict(X_test).flatten()

print('')
print('evaluating...')
score = model.evaluate(X_test, Y_test)
print('')
print('test RMSE: ' + str(np.sqrt(score)))

# if asked, show predictions vs actual on test data
if len(sys.argv) > 1:
    plt.scatter(Y_test, yhat)
    plt.xlabel('actual y')
    plt.ylabel('predicted y')
    plt.show()

    plt.scatter(Y_test, yhat - Y_test)
    plt.xlabel('actual y')
    plt.ylabel('predicted - actual')
    plt.show()
