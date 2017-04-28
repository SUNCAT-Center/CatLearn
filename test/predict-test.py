""" Script to test the prediction functions. """
from __future__ import print_function

import numpy as np

from ase.ga.data import DataConnection
from atoml.data_setup import get_unique, get_train
from atoml.fingerprint_setup import normalize, return_fpv
from atoml.particle_fingerprint import ParticleFingerprintGenerator
from atoml.predict import FitnessPrediction


db = DataConnection('gadb.db')

# Get all relaxed candidates from the db file.
all_cand = db.get_all_relaxed_candidates(use_extinct=False)

# Setup the test and training datasets.
testset = get_unique(candidates=all_cand, testsize=10, key='raw_score')
trainset = get_train(candidates=all_cand, trainsize=50,
                     taken_cand=testset['taken'], key='raw_score')

# Define fingerprint parameters.
fpv = ParticleFingerprintGenerator(get_nl=False, max_bonds=13)

# Get the list of fingerprint vectors and normalize them.
test_fp = return_fpv(testset['candidates'], [fpv.nearestneighbour_fpv])
train_fp = return_fpv(trainset['candidates'], [fpv.nearestneighbour_fpv])
nfp = normalize(train=train_fp, test=test_fp)

# Set up the prediction routine.
kdict = {'k1': {'type': 'linear'}}
krr = FitnessPrediction(kernel_dict=kdict)
#krr = FitnessPrediction(ktype='linear')
#cvm = krr.get_covariance(train_matrix=nfp['train'])
#cinv = np.linalg.inv(cvm)
#assert np.shape(cinv) == (50, 50)
pred = krr.get_predictions(train_fp=nfp['train'],
                           test_fp=nfp['test'],
                           cinv=None,
                           train_target=trainset['target'],
                           test_target=testset['target'],
                           get_validation_error=True,
                           get_training_error=True,
                           optimize_hyperparameters=False)
assert len(pred['prediction']) == 10
print('linear prediction:', pred['validation_rmse']['average'])

# Set up the prediction routine.
kdict = {'k1': {'type': 'polynomial', 'kfree': 0., 'kdegree': 2.}}
krr = FitnessPrediction(kernel_dict=kdict)
#krr = FitnessPrediction(ktype='polynomial',
#                        kfree=0.,
#                        kdegree=2.)
#cvm = krr.get_covariance(train_matrix=nfp['train'])
#cinv = np.linalg.inv(cvm)
#assert np.shape(cinv) == (50, 50)
pred = krr.get_predictions(train_fp=nfp['train'],
                           test_fp=nfp['test'],
                           cinv=None,
                           train_target=trainset['target'],
                           test_target=testset['target'],
                           get_validation_error=True,
                           get_training_error=True,
                           optimize_hyperparameters=False)
assert len(pred['prediction']) == 10
print('polynomial prediction:', pred['validation_rmse']['average'])

# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': 0.5}}
krr = FitnessPrediction(kernel_dict=kdict,
                        regularization=0.001)
#krr = FitnessPrediction(ktype='gaussian',
#                        kwidth=0.5,
#                        regularization=0.001)
#cvm = krr.get_covariance(train_matrix=nfp['train'])
#cinv = np.linalg.inv(cvm)
#assert np.shape(cinv) == (50, 50)
pred = krr.get_predictions(train_fp=nfp['train'],
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

pred = krr.get_predictions(train_fp=nfp['train'],
                           test_fp=nfp['test'],
                           cinv=None,
                           train_target=trainset['target'],
                           test_target=testset['target'],
                           get_validation_error=True,
                           get_training_error=True,
                           cost='absolute')
print('gaussian prediction (abs):', pred['validation_rmse']['average'])

pred = krr.get_predictions(train_fp=nfp['train'],
                           test_fp=nfp['test'],
                           cinv=None,
                           train_target=trainset['target'],
                           test_target=testset['target'],
                           get_validation_error=True,
                           get_training_error=True,
                           cost='insensitive',
                           epsilon=0.1)
print('gaussian prediction (ins):', pred['validation_rmse']['average'])

# Set up the prediction routine.
kdict = {'k1': {'type': 'laplacian', 'width': 0.5}}
krr = FitnessPrediction(kernel_dict=kdict,
                        regularization=0.001)
#krr = FitnessPrediction(ktype='laplacian',
#                        kwidth=0.5,
#                        regularization=0.001)
#cvm = krr.get_covariance(train_matrix=nfp['train'])
#cinv = np.linalg.inv(cvm)
#assert np.shape(cinv) == (50, 50)
pred = krr.get_predictions(train_fp=nfp['train'],
                           test_fp=nfp['test'],
                           cinv=None,
                           train_target=trainset['target'],
                           test_target=testset['target'],
                           get_validation_error=True,
                           get_training_error=True,
                           optimize_hyperparameters=False)
assert len(pred['prediction']) == 10
print('laplacian prediction:', pred['validation_rmse']['average'])

# Set up the prediction routine.
kdict = {'k1': {'type': 'linear', 'features': [0, 1]},
                'k2': {'type': 'gaussian', 'features': [2, 3], 'width': 0.5}
                }
krr = FitnessPrediction(kernel_dict=kdict,
                        regularization=0.001)
#krr = FitnessPrediction(combine_kernels='addition',
#                        kernel_list={'linear': [0, 1], 'gaussian': [2, 3]},
#                        kwidth=0.5,
#                        regularization=0.001)
#cvm = krr.get_covariance(train_matrix=nfp['train'])
#cinv = np.linalg.inv(cvm)
#assert np.shape(cinv) == (50, 50)
pred = krr.get_predictions(train_fp=nfp['train'],
                           test_fp=nfp['test'],
                           cinv=None,
                           train_target=trainset['target'],
                           test_target=testset['target'],
                           get_validation_error=True,
                           get_training_error=True,
                           optimize_hyperparameters=False)
assert len(pred['prediction']) == 10
print('addition prediction:', pred['validation_rmse']['average'])

# Set up the prediction routine.
kdict = {'k1': {'type': 'linear', 'features': [0, 1]},
                'k2': {'type': 'gaussian', 'features': [2, 3], 'width': 0.5,
                       'operation': 'multiplication'}
                }
krr = FitnessPrediction(kernel_dict=kdict,
                        regularization=0.001)
#krr = FitnessPrediction(combine_kernels='multiplication',
#                        kernel_list={'linear': [0, 1], 'gaussian': [2, 3]},
#                        kwidth=0.5,
#                        regularization=0.001)
#cvm = krr.get_covariance(train_matrix=nfp['train'])
#cinv = np.linalg.inv(cvm)
#assert np.shape(cinv) == (50, 50)
pred = krr.get_predictions(train_fp=nfp['train'],
                           test_fp=nfp['test'],
                           cinv=None,
                           train_target=trainset['target'],
                           test_target=testset['target'],
                           get_validation_error=True,
                           get_training_error=True,
                           optimize_hyperparameters=False)
assert len(pred['prediction']) == 10
print('multiplication prediction:', pred['validation_rmse']['average'])
