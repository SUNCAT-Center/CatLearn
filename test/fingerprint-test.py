""" Script to test the fingerprint generation functions. """
from __future__ import print_function

import time
import numpy as np

from ase.ga.data import DataConnection
from atoml.data_setup import get_train, get_unique
from atoml.fingerprint_setup import (return_fpv, sure_independence_screening,
                                     iterative_sis, normalize, standardize)
from atoml.particle_fingerprint import ParticleFingerprintGenerator
from atoml.standard_fingerprint import StandardFingerprintGenerator

db = DataConnection('gadb.db')

# Get all relaxed candidates from the db file.
all_cand = db.get_all_relaxed_candidates(use_extinct=False)

# Setup the test and training datasets.
testset = get_unique(candidates=all_cand, testsize=5, key='raw_score')
trainset = get_train(candidates=all_cand, trainsize=10,
                     taken_cand=testset['taken'], key='raw_score')

# Delete the stored values of nnmat to give better indication of timing.
for i in testset['candidates']:
    del i.info['data']['nnmat']
for i in trainset['candidates']:
    del i.info['data']['nnmat']

# Initiate the fingerprint generators with relevant input variables.
pfpv = ParticleFingerprintGenerator(atom_numbers=[78, 79], max_bonds=13,
                                    get_nl=False, dx=0.2, cell_size=50.,
                                    nbin=4)
sfpv = StandardFingerprintGenerator(atom_types=[78, 79])

# Tests get the list of fingerprint vectors and normalize them.
# Start testing the particle fingerprint vector generators.
start_time = time.time()
test_fp = return_fpv(testset['candidates'], [pfpv.nearestneighbour_fpv],
                     use_prior=False)
train_fp = return_fpv(trainset['candidates'], [pfpv.nearestneighbour_fpv],
                      use_prior=False)
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)
print('nearestneighbour_fpv: %.3f' % (time.time() - start_time))
assert len(test_fp) == 5 and len(train_fp) == 10
assert len(nfp['train'][0]) == 4 and len(nfp['test'][0]) == 4
assert len(sfp['train'][0]) == 4 and len(sfp['test'][0]) == 4

start_time = time.time()
test_fp = return_fpv(testset['candidates'], [pfpv.bond_count_fpv],
                     use_prior=False)
train_fp = return_fpv(trainset['candidates'], [pfpv.bond_count_fpv],
                      use_prior=False)
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)
print('bond_count_fpv: %.3f' % (time.time() - start_time))
assert len(test_fp) == 5 and len(train_fp) == 10
assert len(nfp['train'][0]) == 52 and len(nfp['test'][0]) == 52
assert len(sfp['train'][0]) == 52 and len(sfp['test'][0]) == 52

start_time = time.time()
test_fp = return_fpv(testset['candidates'], [pfpv.distribution_fpv],
                     use_prior=False)
train_fp = return_fpv(trainset['candidates'], [pfpv.distribution_fpv],
                      use_prior=False)
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)
print('distribution_fpv: %.3f' % (time.time() - start_time))
assert len(test_fp) == 5 and len(train_fp) == 10
assert len(nfp['train'][0]) == 8 and len(nfp['test'][0]) == 8
assert len(sfp['train'][0]) == 8 and len(sfp['test'][0]) == 8

start_time = time.time()
test_fp = return_fpv(testset['candidates'], [pfpv.connections_fpv],
                     use_prior=False)
train_fp = return_fpv(trainset['candidates'], [pfpv.connections_fpv],
                      use_prior=False)
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)
print('connections_fpv: %.3f' % (time.time() - start_time))
assert len(test_fp) == 5 and len(train_fp) == 10
assert len(nfp['train'][0]) == 26 and len(nfp['test'][0]) == 26
assert len(sfp['train'][0]) == 26 and len(sfp['test'][0]) == 26

start_time = time.time()
test_fp = return_fpv(testset['candidates'], [pfpv.rdf_fpv], use_prior=False)
train_fp = return_fpv(trainset['candidates'], [pfpv.rdf_fpv], use_prior=False)
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)
print('rdf_fpv: %.3f' % (time.time() - start_time))
assert len(test_fp) == 5 and len(train_fp) == 10
assert len(nfp['train'][0]) == 20 and len(nfp['test'][0]) == 20
assert len(sfp['train'][0]) == 20 and len(sfp['test'][0]) == 20

# Start testing the standard fingerprint vector generators.
start_time = time.time()
test_fp = return_fpv(testset['candidates'], [sfpv.mass_fpv], use_prior=False)
train_fp = return_fpv(trainset['candidates'], [sfpv.mass_fpv], use_prior=False)
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)
print('mass_fpv: %.3f' % (time.time() - start_time))
assert len(test_fp) == 5 and len(train_fp) == 10
assert len(nfp['train'][0]) == 1 and len(nfp['test'][0]) == 1
assert len(sfp['train'][0]) == 1 and len(sfp['test'][0]) == 1

start_time = time.time()
test_fp = return_fpv(testset['candidates'], [sfpv.composition_fpv],
                     use_prior=False)
train_fp = return_fpv(trainset['candidates'], [sfpv.composition_fpv],
                      use_prior=False)
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)
print('composition_fpv: %.3f' % (time.time() - start_time))
assert len(test_fp) == 5 and len(train_fp) == 10
assert len(nfp['train'][0]) == 2 and len(nfp['test'][0]) == 2
assert len(sfp['train'][0]) == 2 and len(sfp['test'][0]) == 2

start_time = time.time()
test_fp = return_fpv(testset['candidates'], [sfpv.eigenspectrum_fpv],
                     use_prior=False)
train_fp = return_fpv(trainset['candidates'], [sfpv.eigenspectrum_fpv],
                      use_prior=False)
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)
print('eigenspectrum_fpv: %.3f' % (time.time() - start_time))
assert len(test_fp) == 5 and len(train_fp) == 10
assert len(nfp['train'][0]) == 147 and len(nfp['test'][0]) == 147
assert len(sfp['train'][0]) == 147 and len(sfp['test'][0]) == 147

start_time = time.time()
test_fp = return_fpv(testset['candidates'], [sfpv.distance_fpv],
                     use_prior=False)
train_fp = return_fpv(trainset['candidates'], [sfpv.distance_fpv],
                      use_prior=False)
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)
print('distance_fpv: %.3f' % (time.time() - start_time))
assert len(test_fp) == 5 and len(train_fp) == 10
assert len(nfp['train'][0]) == 2 and len(nfp['test'][0]) == 2
assert len(sfp['train'][0]) == 2 and len(sfp['test'][0]) == 2

start_time = time.time()
test_fp = return_fpv(testset['candidates'], [pfpv.nearestneighbour_fpv,
                                             sfpv.mass_fpv,
                                             sfpv.composition_fpv])
train_fp = return_fpv(trainset['candidates'], [pfpv.nearestneighbour_fpv,
                                               sfpv.mass_fpv,
                                               sfpv.composition_fpv])

sis = sure_independence_screening(target=trainset['target'],
                                  train_fpv=train_fp, size=4)
sis_test_fp = np.delete(test_fp, sis['rejected'], 1)
sis_train_fp = np.delete(train_fp, sis['rejected'], 1)
assert len(sis_test_fp[0]) == 4 and len(sis_train_fp[0]) == 4

it_sis = iterative_sis(target=trainset['target'], train_fpv=train_fp, size=4,
                       step=1)
it_sis_test_fp = np.delete(test_fp, it_sis['rejected'], 1)
it_sis_train_fp = np.delete(train_fp, it_sis['rejected'], 1)
assert len(sis_test_fp[0]) == 4 and len(sis_train_fp[0]) == 4

nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)
print('Combined features and SIS: %.3f' % (time.time() - start_time))
assert len(test_fp) == 5 and len(train_fp) == 10
assert len(nfp['train'][0]) == 7 and len(nfp['test'][0]) == 7
assert len(sfp['train'][0]) == 7 and len(sfp['test'][0]) == 7
