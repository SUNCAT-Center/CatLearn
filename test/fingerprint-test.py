""" Script to test the fingerprint generation functions. """
from ase.ga.data import DataConnection
from predict.data_setup import get_train, get_unique
from predict.fingerprint_setup import (normalize, standardize, get_single_fpv,
                                       get_combined_fpv)
from predict.particle_fingerprint import ParticleFingerprintGenerator
from predict.standard_fingerprint import StandardFingerprintGenerator

db = DataConnection('gadb.db')

# Get all relaxed candidates from the db file.
all_cand = db.get_all_relaxed_candidates(use_extinct=False)

# Setup the test and training datasets.
testset = get_unique(candidates=all_cand, testsize=5)
trainset = get_train(candidates=all_cand, trainsize=10,
                     taken_cand=testset['taken'], key='raw_score')

pfpv = ParticleFingerprintGenerator(atom_numbers=[78, 79], max_bonds=13,
                                    get_nl=False, dx=0.2, cell_size=50.,
                                    nbin=4)
sfpv = StandardFingerprintGenerator(atom_types=['Pt', 'Au'])

# Get the list of fingerprint vectors and normalize them.
# Start testing the particle fingerprint vector generators.
test_fp = get_single_fpv(testset['candidates'], pfpv.nearestneighbour_fpv)
train_fp = get_single_fpv(trainset['candidates'], pfpv.nearestneighbour_fpv)
assert len(test_fp) == 5 and len(train_fp) == 10
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)

test_fp = get_single_fpv(testset['candidates'], pfpv.bond_count_fpv)
train_fp = get_single_fpv(trainset['candidates'], pfpv.bond_count_fpv)
assert len(test_fp) == 5 and len(train_fp) == 10
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)

test_fp = get_single_fpv(testset['candidates'], pfpv.distribution_fpv)
train_fp = get_single_fpv(trainset['candidates'], pfpv.distribution_fpv)
assert len(test_fp) == 5 and len(train_fp) == 10
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)

test_fp = get_single_fpv(testset['candidates'], pfpv.connections_fpv)
train_fp = get_single_fpv(trainset['candidates'], pfpv.connections_fpv)
assert len(test_fp) == 5 and len(train_fp) == 10
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)

test_fp = get_single_fpv(testset['candidates'], pfpv.rdf_fpv)
train_fp = get_single_fpv(trainset['candidates'], pfpv.rdf_fpv)
assert len(test_fp) == 5 and len(train_fp) == 10
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)

# Start testing the standard fingerprint vector generators.
test_fp = get_single_fpv(testset['candidates'], sfpv.mass_fpv)
train_fp = get_single_fpv(trainset['candidates'], sfpv.mass_fpv)
assert len(test_fp) == 5 and len(train_fp) == 10
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)

test_fp = get_single_fpv(testset['candidates'], sfpv.composition_fpv)
train_fp = get_single_fpv(trainset['candidates'], sfpv.composition_fpv)
assert len(test_fp) == 5 and len(train_fp) == 10
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)

test_fp = get_single_fpv(testset['candidates'], sfpv.eigenspectrum_fpv)
train_fp = get_single_fpv(trainset['candidates'], sfpv.eigenspectrum_fpv)
assert len(test_fp) == 5 and len(train_fp) == 10
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)

test_fp = get_single_fpv(testset['candidates'], sfpv.distance_fpv)
train_fp = get_single_fpv(trainset['candidates'], sfpv.distance_fpv)
assert len(test_fp) == 5 and len(train_fp) == 10
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)

test_fp = get_combined_fpv(testset['candidates'], [pfpv.nearestneighbour_fpv,
                                                   sfpv.mass_fpv,
                                                   sfpv.composition_fpv])
train_fp = get_combined_fpv(trainset['candidates'], [pfpv.nearestneighbour_fpv,
                                                     sfpv.mass_fpv,
                                                     sfpv.composition_fpv])
assert len(test_fp) == 5 and len(train_fp) == 10
nfp = normalize(test=test_fp, train=train_fp)
sfp = standardize(test=test_fp, train=train_fp)
