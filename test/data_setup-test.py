"""Script to test the data_setup functions."""
from __future__ import print_function
from __future__ import absolute_import

from ase.ga.data import DataConnection
from atoml.data_setup import get_unique, get_train
from atoml.utilities import remove_outliers

db = DataConnection('../data/gadb.db')

# Get all relaxed candidates from the db file.
all_cand = db.get_all_relaxed_candidates(use_extinct=False)
prune = remove_outliers(all_cand, key='raw_score')
assert len(all_cand) != len(prune)

# Setup the test and training datasets.
testset = get_unique(atoms=all_cand, size=10, key='raw_score')
assert len(testset['atoms']) == 10
assert len(testset['taken']) == 10

trainset = get_train(atoms=all_cand, size=50, taken=testset['taken'],
                     key='raw_score')
assert len(trainset['atoms']) == 50
assert len(trainset['target']) == 50
