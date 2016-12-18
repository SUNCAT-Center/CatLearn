""" Script to test the ML model. Takes a database of candidates from a GA
    search with a raw_score set in atoms.info['key_value_pairs'][key] and
    returns the errors for a random test and training dataset.
"""
from __future__ import print_function

from ase.ga.data import DataConnection
from predict.data_setup import get_unique, get_train
from predict.fingerprint_setup import normalize, get_single_fpv
from predict.particle_fingerprint import ParticleFingerprintGenerator
from predict.predict import FitnessPrediction


db = DataConnection('gadb.db')

# Get all relaxed candidates from the db file.
print('Getting candidates from the database')
all_cand = db.get_all_relaxed_candidates(use_extinct=False)

# Setup the test and training datasets.
testset = get_unique(candidates=all_cand, testsize=500)
trainset = get_train(candidates=all_cand, trainsize=500,
                     taken_cand=testset['taken'], key='raw_score')

# Get the list of fingerprint vectors and normalize them.
print('Getting the fingerprint vectors')
fpv = ParticleFingerprintGenerator(get_nl=False, max_bonds=13)
test_fp = get_single_fpv(testset['candidates'], fpv.bond_count_fpv)
train_fp = get_single_fpv(trainset['candidates'], fpv.bond_count_fpv)
nfp = normalize(train=train_fp, test=test_fp)

# Set up the prediction routine.
krr = FitnessPrediction(ktype='gaussian',
                        kwidth=0.5,
                        regularization=0.001)

# Do the predictions.
cvm = krr.get_covariance(train_fp=nfp['train'])
print('Making the predictions')
pred = krr.get_predictions(train_fp=nfp['train'],
                           test_fp=nfp['test'],
                           cinv=cvm,
                           target=trainset['target'],
                           known=True,
                           test=testset['candidates'],
                           key='raw_score')

# Print the error associated with the predictions.
print('Model error:', pred['rmse'])
