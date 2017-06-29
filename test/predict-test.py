"""Script to test the prediction functions."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from ase.ga.data import DataConnection
from atoml.data_setup import get_unique, get_train
from atoml.fingerprint_setup import return_fpv
from atoml.feature_preprocess import normalize
from atoml.particle_fingerprint import ParticleFingerprintGenerator
from atoml.predict import GaussianProcess


db = DataConnection('../data/gadb.db')

# Get all relaxed candidates from the db file.
all_cand = db.get_all_relaxed_candidates(use_extinct=False)

# Setup the test and training datasets.
testset = get_unique(atoms=all_cand, size=10, key='raw_score')
trainset = get_train(atoms=all_cand, size=50, taken=testset['taken'],
                     key='raw_score')

# Define fingerprint parameters.
fpv = ParticleFingerprintGenerator(get_nl=False, max_bonds=13)

# Get the list of fingerprint vectors and normalize them.
test_fp = return_fpv(testset['atoms'], [fpv.nearestneighbour_fpv])
train_fp = return_fpv(trainset['atoms'], [fpv.nearestneighbour_fpv])
nfp = normalize(train_matrix=train_fp, test_matrix=test_fp)

# Test prediction routine with linear kernel.
kdict = {'k1': {'type': 'linear', 'const': 0.}}
gp = GaussianProcess(kernel_dict=kdict)
pred = gp.get_predictions(train_fp=nfp['train'],
                          test_fp=nfp['test'],
                          cinv=None,
                          train_target=trainset['target'],
                          test_target=testset['target'],
                          get_validation_error=True,
                          get_training_error=True,
                          optimize_hyperparameters=False)
assert len(pred['prediction']) == 10
print('linear prediction:', pred['validation_rmse']['average'])

# Test prediction routine with polynomial kernel.
kdict = {'k1': {'type': 'polynomial', 'slope': 0.5, 'degree': 2., 'const': 0.}}
gp = GaussianProcess(kernel_dict=kdict)
pred = gp.get_predictions(train_fp=nfp['train'],
                          test_fp=nfp['test'],
                          cinv=None,
                          train_target=trainset['target'],
                          test_target=testset['target'],
                          get_validation_error=True,
                          get_training_error=True,
                          optimize_hyperparameters=False)
assert len(pred['prediction']) == 10
print('polynomial prediction:', pred['validation_rmse']['average'])

# Test prediction routine with gaussian kernel.
kdict = {'k1': {'type': 'gaussian', 'width': 0.5}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)
pred = gp.get_predictions(train_fp=nfp['train'],
                          test_fp=nfp['test'],
                          cinv=None,
                          train_target=trainset['target'],
                          test_target=testset['target'],
                          get_validation_error=True,
                          get_training_error=True,
                          uncertainty=True,
                          cost='squared')
assert len(pred['prediction']) == 10
print('gaussian prediction:', pred['validation_rmse']['average'])
for i, j, k, in zip(pred['prediction'],
                    pred['uncertainty'],
                    pred['validation_rmse']['all']):
    print(i, j, k)

pred = gp.get_predictions(train_fp=nfp['train'],
                          test_fp=nfp['test'],
                          cinv=None,
                          train_target=trainset['target'],
                          test_target=testset['target'],
                          get_validation_error=True,
                          get_training_error=True,
                          cost='absolute')
print('gaussian prediction (abs):', pred['validation_rmse']['average'])

pred = gp.get_predictions(train_fp=nfp['train'],
                          test_fp=nfp['test'],
                          cinv=None,
                          train_target=trainset['target'],
                          test_target=testset['target'],
                          get_validation_error=True,
                          get_training_error=True,
                          cost='insensitive',
                          epsilon=0.1)
print('gaussian prediction (ins):', pred['validation_rmse']['average'])

# Test prediction routine with laplacian kernel.
kdict = {'k1': {'type': 'laplacian', 'width': 0.5}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)
pred = gp.get_predictions(train_fp=nfp['train'],
                          test_fp=nfp['test'],
                          cinv=None,
                          train_target=trainset['target'],
                          test_target=testset['target'],
                          get_validation_error=True,
                          get_training_error=True,
                          optimize_hyperparameters=True)
assert len(pred['prediction']) == 10
print('laplacian prediction:', pred['validation_rmse']['average'])

# Prepare discrete data
discrete_train = np.digitize(nfp['train'], np.linspace(-1, 1, 8))
discrete_test = np.digitize(nfp['test'], np.linspace(-1, 1, 8))
# Prepare hyperparameters for AA kernel
cAA = []
for column in range(np.shape(discrete_train)[1]):
    cAA.append(len(np.unique(discrete_train[:, column])))
# Test prediction routine with AA kernel.
kdict = {'k1': {'type': 'AA', 'theta': [.75] + cAA}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)
pred = gp.get_predictions(train_fp=discrete_train,
                          test_fp=discrete_test,
                          cinv=None,
                          train_target=trainset['target'],
                          test_target=testset['target'],
                          get_validation_error=True,
                          get_training_error=True,
                          optimize_hyperparameters=False)
assert len(pred['prediction']) == 10
print('AA prediction:', pred['validation_rmse']['average'])

# Test prediction routine with addative linear and gaussian kernel.
kdict = {'k1': {'type': 'linear', 'features': [0, 1], 'const': 0.},
         'k2': {'type': 'gaussian', 'features': [2, 3], 'width': 0.5}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)
pred = gp.get_predictions(train_fp=nfp['train'],
                          test_fp=nfp['test'],
                          cinv=None,
                          train_target=trainset['target'],
                          test_target=testset['target'],
                          get_validation_error=True,
                          get_training_error=True,
                          optimize_hyperparameters=True)
assert len(pred['prediction']) == 10
print('addition prediction:', pred['validation_rmse']['average'])

# Test prediction routine with multiplication of linear and gaussian kernel.
kdict = {'k1': {'type': 'linear', 'features': [0, 1], 'const': 0.},
         'k2': {'type': 'gaussian', 'features': [2, 3], 'width': 0.5,
                'operation': 'multiplication'}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)
pred = gp.get_predictions(train_fp=nfp['train'],
                          test_fp=nfp['test'],
                          cinv=None,
                          train_target=trainset['target'],
                          test_target=testset['target'],
                          get_validation_error=True,
                          get_training_error=True,
                          optimize_hyperparameters=True)
assert len(pred['prediction']) == 10
print('multiplication prediction:', pred['validation_rmse']['average'])
