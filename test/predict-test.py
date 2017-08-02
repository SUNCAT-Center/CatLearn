"""Script to test the prediction functions."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from atoml.database_functions import DescriptorDatabase
from atoml.feature_preprocess import standardize, normalize
from atoml.ridge_regression import RidgeRegression
from atoml.predict import GaussianProcess

# Attach the database.
dd = DescriptorDatabase(db_name='fpv_store.sqlite',
                        table='FingerVector')

# Pull the features and targets from the database.
names = dd.get_column_names()
features, targets = names[1:-1], names[-1:]
feature_data = dd.query_db(names=features)
target_data = np.reshape(dd.query_db(names=targets),
                         (np.shape(feature_data)[0], ))

# Split the data into so test and training sets.
train_features, train_targets = feature_data[:35, :], target_data[:35]
test_features, test_targets = feature_data[35:, :], target_data[35:]

# Test ridge regression predictions.
rr = RidgeRegression()
reg = rr.find_optimal_regularization(X=train_features, Y=train_targets)
coef = rr.RR(X=train_features, Y=train_targets, omega2=reg)[0]

# Test the model.
sumd = 0.
for tf, tt in zip(test_features, test_targets):
    p = (np.dot(coef, tf))
    sumd += (p - tt) ** 2
print('Ridge regression prediction:', (sumd / len(test_features)) ** 0.5)

sfp = standardize(train_matrix=train_features, test_matrix=test_features)
nfp = normalize(train_matrix=train_features, test_matrix=test_features)

# Test prediction routine with linear kernel.
kdict = {'k1': {'type': 'linear', 'const': 0.}}
gp = GaussianProcess(train_fp=sfp['train'], train_target=train_targets,
                     kernel_dict=kdict, regularization=0.001,
                     optimize_hyperparameters=False)
pred = gp.get_predictions(test_fp=sfp['test'],
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)
assert len(pred['prediction']) == len(sfp['test'])
print('linear prediction:', pred['validation_error']['rmse_average'])

# Test prediction routine with polynomial kernel.
kdict = {'k1': {'type': 'polynomial', 'slope': 0.5, 'degree': 2., 'const': 0.}}
gp = GaussianProcess(train_fp=sfp['train'], train_target=train_targets,
                     kernel_dict=kdict, regularization=0.001,
                     optimize_hyperparameters=False)
pred = gp.get_predictions(test_fp=sfp['test'],
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)
assert len(pred['prediction']) == len(sfp['test'])
print('polynomial prediction:', pred['validation_error']['rmse_average'])

# Test prediction routine with gaussian kernel.
kdict = {'k1': {'type': 'gaussian', 'width': 0.5}}
gp = GaussianProcess(train_fp=sfp['train'], train_target=train_targets,
                     kernel_dict=kdict, regularization=0.001,
                     optimize_hyperparameters=False)
pred = gp.get_predictions(test_fp=sfp['test'],
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True,
                          uncertainty=True,
                          epsilon=0.1)
assert len(pred['prediction']) == len(sfp['test'])
print('gaussian prediction (rmse):', pred['validation_error']['rmse_average'])
for i, j, k, in zip(pred['prediction'],
                    pred['uncertainty'],
                    pred['validation_error']['rmse_all']):
    print(i, j, k)

print('gaussian prediction (ins):',
      pred['validation_error']['insensitive_average'])
print('gaussian prediction (abs):',
      pred['validation_error']['absolute_average'])

# Test prediction routine with laplacian kernel.
kdict = {'k1': {'type': 'laplacian', 'width': 0.5}}
gp = GaussianProcess(train_fp=sfp['train'], train_target=train_targets,
                     kernel_dict=kdict, regularization=0.001,
                     optimize_hyperparameters=True)
pred = gp.get_predictions(test_fp=sfp['test'],
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)
assert len(pred['prediction']) == len(sfp['test'])
print('laplacian prediction:', pred['validation_error']['rmse_average'])

# Test prediction routine with addative linear and gaussian kernel.
kdict = {'k1': {'type': 'linear', 'features': [0, 1], 'const': 0.},
         'k2': {'type': 'gaussian', 'features': [2, 3], 'width': 0.5}}
gp = GaussianProcess(train_fp=sfp['train'], train_target=train_targets,
                     kernel_dict=kdict, regularization=0.001,
                     optimize_hyperparameters=True)
pred = gp.get_predictions(test_fp=sfp['test'],
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)
assert len(pred['prediction']) == len(sfp['test'])
print('addition prediction:', pred['validation_error']['rmse_average'])

# Test prediction routine with multiplication of linear and gaussian kernel.
kdict = {'k1': {'type': 'linear', 'features': [0, 1], 'const': 0.},
         'k2': {'type': 'gaussian', 'features': [2, 3], 'width': 0.5,
                'operation': 'multiplication'}}
gp = GaussianProcess(train_fp=sfp['train'], train_target=train_targets,
                     kernel_dict=kdict, regularization=0.001,
                     optimize_hyperparameters=True)
pred = gp.get_predictions(test_fp=sfp['test'],
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)
assert len(pred['prediction']) == len(sfp['test'])
print('multiplication prediction:', pred['validation_error']['rmse_average'])
