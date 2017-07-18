"""Script to test descriptors for the ML model."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import time

from atoml.cross_validation import HierarchyValidation
from atoml.feature_preprocess import standardize
from atoml.feature_engineering import (single_transform, get_order_2,
                                       get_order_2ab, get_ablog,
                                       get_div_order_2)
from atoml.feature_elimination import FeatureScreening
from atoml.predict import GaussianProcess

# Set some parameters.
plot = False
ds = 500

# Define the hierarchey cv class method.
hv = HierarchyValidation(db_name='../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='split')
# Split the data into subsets.
hv.split_index(min_split=ds, max_split=ds*2)
# Load data back in from save file.
ind = hv.load_split()

# Split out the various data.
train_data = np.array(hv._compile_split(ind['1_1'])[:, 1:-1], np.float64)
train_target = np.array(hv._compile_split(ind['1_1'])[:, -1:], np.float64)
test_data = np.array(hv._compile_split(ind['1_2'])[:, 1:-1], np.float64)
test_target = np.array(hv._compile_split(ind['1_2'])[:, -1:], np.float64)

d, f = np.shape(train_data)

# Scale and shape the data.
std = standardize(train_matrix=train_data, test_matrix=test_data)
train_data, test_data = std['train'], std['test']
train_target = train_target.reshape(len(train_target), )
test_target = test_target.reshape(len(test_target), )

# Expand feature space to add single variable transforms.
test_data1 = single_transform(test_data)
train_data1 = single_transform(train_data)
test_data2 = get_order_2(test_data)
train_data2 = get_order_2(train_data)
# test_data3 = get_div_order_2(test_data)
# train_data3 = get_div_order_2(train_data)
# test_data4 = get_order_2ab(test_data, 2, 3)
# train_data4 = get_order_2ab(train_data, 2, 3)
# test_data5 = get_ablog(test_data, 2, 3)
# train_data5 = get_ablog(train_data, 2, 3)

test_data = np.concatenate((test_data, test_data1), axis=1)
train_data = np.concatenate((train_data, train_data1), axis=1)
test_data = np.concatenate((test_data, test_data2), axis=1)
train_data = np.concatenate((train_data, train_data2), axis=1)
# test_data = np.concatenate((test_data, test_data3), axis=1)
# train_data = np.concatenate((train_data, train_data3), axis=1)
# test_data = np.concatenate((test_data, test_data4), axis=1)
# train_data = np.concatenate((train_data, train_data4), axis=1)
# test_data = np.concatenate((test_data, test_data5), axis=1)
# train_data = np.concatenate((train_data, train_data5), axis=1)


def do_pred(train, test):
    """Function to make prediction."""
    pred = gp.get_predictions(train_fp=train,
                              test_fp=test,
                              train_target=train_target,
                              test_target=test_target,
                              get_validation_error=True,
                              get_training_error=True,
                              optimize_hyperparameters=True)

    # Print the error associated with the predictions.
    print('Training error:', pred['training_error']['rmse_average'])
    print('Model error:', pred['validation_error']['rmse_average'])


# Get base predictions.
print('\nBase Predictions\n')
# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': 10.}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)
do_pred(train=train_data[:, :f], test=test_data[:, :f])

# Get descriptor correlation
corr = ['pearson', 'spearman', 'kendall']
for c in corr:
    print('\nPredictions based on %s correlation\n' % c)
    # Set up the prediction routine.
    kdict = {'k1': {'type': 'gaussian', 'width': 10.}}
    gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)

    screen = FeatureScreening(correlation=c, iterative=False)
    st = time.time()
    features = screen.eliminate_features(target=train_target,
                                         train_features=train_data,
                                         test_features=test_data,
                                         size=d, step=None, order=None)
    print('screening took:', time.time() - st, 'for', np.shape(train_data))
    reduced_train = features[0]
    reduced_test = features[1]
    do_pred(train=reduced_train, test=reduced_test)

    # Set up the prediction routine.
    kdict = {'k1': {'type': 'gaussian', 'width': 10.}}
    gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)

    screen = FeatureScreening(correlation=c, iterative=True,
                              regression='lasso')
    st = time.time()
    features = screen.eliminate_features(target=train_target,
                                         train_features=train_data,
                                         test_features=test_data,
                                         size=d, step=None, order=None)
    print('iterative took:', time.time() - st, 'for', np.shape(train_data))
    reduced_train = features[0]
    reduced_test = features[1]
    do_pred(train=reduced_train, test=reduced_test)
