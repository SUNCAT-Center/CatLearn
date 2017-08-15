"""Variable sensitivity testing with TensorFlow."""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
from keras.layers.normalization import BatchNormalization as BatchNorm
from keras.optimizers import Adam
from keras.utils import plot_model

from atoml.predict import GaussianProcess
from atoml.cross_validation import HierarchyValidation
from atoml.feature_engineering import single_transform
from atoml.feature_preprocess import standardize

new_data = False
expand = False
plot = True
pm = False
test_train = False
stack_data = 100  # stack training data with itself
size = 500
reg = 0.01

# Define the hierarchey cv class method.
hv = HierarchyValidation(db_name='../../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='sensitivity_data')

# Split the data into subsets.
if new_data:
    hv.split_index(min_split=size, max_split=size*2)
# Load data back in from save file.
ind = hv.load_split()

train_data = np.array(hv._compile_split(ind['1_1'])[:, 1:-1], np.float64)
train_target = np.array(hv._compile_split(ind['1_1'])[:, -1:], np.float64)
test_data = np.array(hv._compile_split(ind['1_2'])[:, 1:-1], np.float64)
test_target = np.array(hv._compile_split(ind['1_2'])[:, -1:], np.float64)

data = np.concatenate((train_data, test_data), axis=0)
data = standardize(train_matrix=data)['train']
if expand:
    # Expand feature space to add single variable transforms.
    data = np.concatenate((data, single_transform(data)), axis=1)
    data = standardize(train_matrix=data)['train']

feat = np.array(range(np.shape(data)[1]))
ti = np.reshape(feat, (1, len(feat)))
data = np.concatenate((ti, data), axis=0)

# np.random.shuffle(np.transpose(data))

feat = np.reshape(data[:1, :], (np.shape(feat)[0],))
data = data[1:, :]

# build training and test sets
X_train = data[:size, :]
Y_train = np.reshape(train_target, (np.shape(train_target)[0],))
X_test = data[-size:, :]
Y_test = np.reshape(test_target, (np.shape(test_target)[0],))

train_num = np.shape(X_train)[0]

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
model.add(Dense(48, input_dim=X_train.shape[1],
                kernel_regularizer=regularizers.l1(reg)))
model.add(BatchNorm())
model.add(Activation('relu'))
# add output layer
model.add(Dense(1))

# TRAIN NEURAL NETWORK

# define optimizer (can tune optimizer parameters, e.g. learning rate)
optimizer = Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='mse')

# Workaround for bug in current version of keras, needed with BatchNorm layers
backend.get_session().run(tf.global_variables_initializer())

# train the model
model.fit(X_train, Y_train, epochs=20, verbose=1, batch_size=1024)

if pm:
    plot_model(model, show_shapes=True, to_file='model.png')

# EVALUATE

pres1 = []
nres1 = []
pres5 = []
nres5 = []
pres25 = []
nres25 = []

# predictions on test set
if test_train:
    X_test = X_train
    Y_test = Y_train
base = model.evaluate(X_test, Y_test)
print('TF Model error:', base)


def sensitivity(val, res):
    """Function to find variable sensitivity."""
    for i in range(1, np.shape(X_test)[1]+1):
        X_now = np.copy(X_test)
        X_now[:, i-1:i] = val
        score = model.evaluate(X_now, Y_test)
        res.append(np.abs(np.sqrt(score) - np.sqrt(base)))


# predictions on test set
res_list = [pres1, nres1, pres5, nres5, pres25, nres25]
res_val = [1., -1., .5, -.5, .25, -.25]

for i, j in zip(res_list, res_val):
    sensitivity(val=j, res=i)

ave = (np.array(pres1) + np.array(nres1) + np.array(pres5) + np.array(nres5)
       + np.array(pres25) + np.array(nres25))
ave /= 6

# Print out a subset of the best features.
print('\n', np.argsort(ave)[-20:])


def do_predict(train, test, train_target, test_target, width, reg, hopt=False):
    """Function to make predictions."""
    kdict = {'k1': {'type': 'gaussian', 'width': width}}
    gp = GaussianProcess(train_fp=train, train_target=train_target,
                         kernel_dict=kdict, regularization=reg,
                         optimize_hyperparameters=hopt)
    # Do the predictions.
    pred = gp.get_predictions(test_fp=test,
                              test_target=test_target,
                              get_validation_error=True,
                              get_training_error=True)
    w = gp.kernel_dict['k1']['width']
    r = gp.regularization

    return pred, w, r


# reform the training target values
Y_train = np.reshape(train_target, (np.shape(train_target)[0],))

w = 10.
width = [w] * np.shape(X_train)[1]
reg = 1e-4
# build training and test sets
X_train = data[:size, :]
X_test = data[-size:, :]

print(np.shape(X_train), np.shape(X_test), np.shape(Y_train), np.shape(Y_test))

print('Optimized all parameters')
a = do_predict(train=X_train, test=X_test, train_target=Y_train,
               test_target=Y_test, hopt=True, width=width, reg=reg)
be = a[0]['validation_error']['rmse_average']

print(list(a[1]), a[2])

# Print the error associated with the predictions.
print('\nGP full model error:', a[0]['validation_error']['rmse_average'], '\n')

err = []

for i in range(100, 110):
    # if width is None:
    width = [w] * i
    reg = 1e-4
    # build training and test sets
    X_train = data[:size, :]
    X_test = data[-size:, :]

    X_train = np.delete(X_train, np.argsort(ave)[:-i], axis=1)
    X_test = np.delete(X_test, np.argsort(ave)[:-i], axis=1)

    print(np.shape(X_train), np.shape(X_test), np.shape(Y_train),
          np.shape(Y_test))

    print('Optimized parameters')
    a = do_predict(train=X_train, test=X_test, train_target=Y_train,
                   test_target=Y_test, hopt=True, width=width, reg=reg)
    err.append(a[0]['validation_error']['rmse_average'] - be)

    print(list(a[1]), a[2])

    # Print the error associated with the predictions.
    print('\nGP Model error:', a[0]['validation_error']['rmse_average'], '\n')

ind = [i for i in range(len(err))]

if plot:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(121)
    ax.scatter(ind, err)
    plt.xlabel('feature')
    plt.ylabel('response')

    ax1 = fig.add_subplot(122)
    ax1.scatter(x=feat, y=pres1, label='+1.00', alpha=0.1)
    ax1.scatter(x=feat, y=pres5, label='+0.50', alpha=0.1)
    ax1.scatter(x=feat, y=pres25, label='+0.25', alpha=0.1)
    ax1.scatter(x=feat, y=nres25, label='-0.25', alpha=0.1)
    ax1.scatter(x=feat, y=nres5, label='-0.50', alpha=0.1)
    ax1.scatter(x=feat, y=nres1, label='-1.00', alpha=0.1)
    ax1.scatter(x=feat, y=ave, label='average', alpha=0.8)
    plt.xlabel('feature')
    plt.ylabel('response')
    plt.legend(loc='upper left')
    plt.show()
