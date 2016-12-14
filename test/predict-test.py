""" Script to test the prediction functions. """
from __future__ import print_function

import numpy as np

from ase.ga.data import DataConnection
from predict.data_setup import get_unique, get_train
from predict.fingerprint_setup import normalize, get_single_fpv
from predict.particle_fingerprint import ParticleFingerprintGenerator
from predict.predict import FitnessPrediction


db = DataConnection('gadb.db')

# Get all relaxed candidates from the db file.
all_cand = db.get_all_relaxed_candidates(use_extinct=False)

# Setup the test and training datasets.
testset = get_unique(candidates=all_cand, testsize=10)
trainset = get_train(candidates=all_cand, trainsize=50,
                     taken_cand=testset['taken'], key='raw_score')

# Define fingerprint parameters.
fpv = ParticleFingerprintGenerator(get_nl=False, max_bonds=13)

# Get the list of fingerprint vectors and normalize them.
test_fp = get_single_fpv(testset['candidates'], fpv.nearestneighbour_fpv)
train_fp = get_single_fpv(trainset['candidates'], fpv.nearestneighbour_fpv)
nfp = normalize(train=train_fp, test=test_fp)

# Set up the prediction routine.
krr = FitnessPrediction(ktype='linear')
cvm = krr.get_covariance(train_fp=nfp['train'])
assert np.shape(cvm) == (50, 50)
pred = krr.get_predictions(train_fp=nfp['train'],
                           test_fp=nfp['test'],
                           cinv=cvm,
                           target=trainset['target'],
                           known=True,
                           test=testset['candidates'],
                           key='raw_score')
assert len(pred['prediction']) == 10
print('linear prediction:', pred['rmse'])

# Set up the prediction routine.
krr = FitnessPrediction(ktype='polynomial',
                        kfree=0.,
                        kdegree=2.)
cvm = krr.get_covariance(train_fp=nfp['train'])
assert np.shape(cvm) == (50, 50)
pred = krr.get_predictions(train_fp=nfp['train'],
                           test_fp=nfp['test'],
                           cinv=cvm,
                           target=trainset['target'],
                           known=True,
                           test=testset['candidates'],
                           key='raw_score')
assert len(pred['prediction']) == 10
print('polynomial prediction:', pred['rmse'])

# Set up the prediction routine.
krr = FitnessPrediction(ktype='gaussian',
                        kwidth=0.5,
                        regularization=0.001)
cvm = krr.get_covariance(train_fp=nfp['train'])
assert np.shape(cvm) == (50, 50)
pred = krr.get_predictions(train_fp=nfp['train'],
                           test_fp=nfp['test'],
                           cinv=cvm,
                           target=trainset['target'],
                           known=True,
                           test=testset['candidates'],
                           key='raw_score')
assert len(pred['prediction']) == 10
print('gaussian prediction:', pred['rmse'])

# Set up the prediction routine.
krr = FitnessPrediction(ktype='laplacian',
                        kwidth=0.5,
                        regularization=0.001)
cvm = krr.get_covariance(train_fp=nfp['train'])
assert np.shape(cvm) == (50, 50)
pred = krr.get_predictions(train_fp=nfp['train'],
                           test_fp=nfp['test'],
                           cinv=cvm,
                           target=trainset['target'],
                           known=True,
                           test=testset['candidates'],
                           key='raw_score')
assert len(pred['prediction']) == 10
print('laplacian prediction:', pred['rmse'])
