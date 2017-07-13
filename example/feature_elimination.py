"""Script to test descriptors for the ML model."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from atoml.cross_validation import HierarchyValidation
from atoml.feature_preprocess import standardize
from atoml.feature_engineering import single_transform
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
test_data = np.concatenate((test_data, single_transform(test_data)), axis=1)
train_data = np.concatenate((train_data, single_transform(train_data)), axis=1)


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
    print('Training error:', pred['training_rmse']['average'])
    print('Model error:', pred['validation_rmse']['average'])


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
    features = screen.eliminate_features(target=train_target,
                                         train_features=train_data,
                                         test_features=test_data,
                                         size=d, step=None, order=None)
    reduced_train = features[0]
    reduced_test = features[1]
    do_pred(train=reduced_train, test=reduced_test)

    # Set up the prediction routine.
    kdict = {'k1': {'type': 'gaussian', 'width': 10.}}
    gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)

    screen = FeatureScreening(correlation=c, iterative=True,
                              regression='lasso')
    features = screen.eliminate_features(target=train_target,
                                         train_features=train_data,
                                         test_features=test_data,
                                         size=d, step=None, order=None)
    reduced_train = features[0]
    reduced_test = features[1]
    do_pred(train=reduced_train, test=reduced_test)
