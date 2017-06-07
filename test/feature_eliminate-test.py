"""Script to test the fingerprint generation functions."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from ase.ga.data import DataConnection
from atoml.data_setup import get_train, get_unique
from atoml.fingerprint_setup import return_fpv
from atoml.feature_elimination import FeatureScreening
from atoml.particle_fingerprint import ParticleFingerprintGenerator
from atoml.standard_fingerprint import StandardFingerprintGenerator

db = DataConnection('gadb.db')

# Get all relaxed candidates from the db file.
all_cand = db.get_all_relaxed_candidates(use_extinct=False)

# Setup the test and training datasets.
testset = get_unique(atoms=all_cand, size=5, key='raw_score')
trainset = get_train(atoms=all_cand, size=10, taken=testset['taken'],
                     key='raw_score')

# Delete the stored values of nnmat to give better indication of timing.
for i in testset['atoms']:
    del i.info['data']['nnmat']
for i in trainset['atoms']:
    del i.info['data']['nnmat']

# Initiate the fingerprint generators with relevant input variables.
pfpv = ParticleFingerprintGenerator(atom_numbers=[78, 79], max_bonds=13,
                                    get_nl=False, dx=0.2, cell_size=50.,
                                    nbin=4)
sfpv = StandardFingerprintGenerator(atom_types=[78, 79])

test_features = return_fpv(testset['atoms'], [pfpv.nearestneighbour_fpv,
                                              pfpv.bond_count_fpv,
                                              sfpv.mass_fpv,
                                              sfpv.composition_fpv,
                                              sfpv.distance_fpv])
train_features = return_fpv(trainset['atoms'], [pfpv.nearestneighbour_fpv,
                                                pfpv.bond_count_fpv,
                                                sfpv.mass_fpv,
                                                sfpv.composition_fpv,
                                                sfpv.distance_fpv])

# Get descriptor correlation
corr = ['pearson', 'spearman', 'kendall']
for c in corr:
    screen = FeatureScreening(correlation=c, iterative=False)
    features = screen.eliminate_features(target=trainset['target'],
                                         train_features=train_features,
                                         test_features=test_features,
                                         size=4, step=None, order=None)
    assert np.shape(features[0])[1] == 4 and np.shape(features[1])[1] == 4

    screen = FeatureScreening(correlation=c, iterative=True,
                              regression='ridge')
    features = screen.eliminate_features(target=trainset['target'],
                                         train_features=train_features,
                                         test_features=test_features,
                                         size=4, step=2, order=None)
    assert np.shape(features[0])[1] == 4 and np.shape(features[1])[1] == 4
