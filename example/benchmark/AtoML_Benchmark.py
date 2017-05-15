import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

from atoml.predict import GaussianProcess
# from atoml.fingerprint_setup import normalize

# load and set up data
shuffle = False  # whether to randomize the order of data
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
# train_num = int(.75 * N)  # how much data for training
# remove = int(0.0 * N)  # how much data to excerpt between train and test
# test_num = N - train_num - remove
train_num = 14000
test_num = 5000
X_train = X[:train_num, :]
Y_train = Y[:train_num]
X_test = X[-test_num:, :]
Y_test = Y[-test_num:]

print(np.shape(X_train), np.shape(X_test), np.shape(Y_train))

# nfp = normalize(train=X_train, test=X_test)

# Test prediction routine with gaussian kernel.
kdict = {'k1': {'type': 'gaussian', 'width': 0.5}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)
pred = gp.get_predictions(train_fp=X_train,
                          test_fp=X_test,
                          train_target=Y_train,
                          test_target=Y_test,
                          get_validation_error=True,
                          get_training_error=True,
                          uncertainty=False,
                          cost='squared')

print('gaussian prediction:', pred['validation_rmse']['average'])
