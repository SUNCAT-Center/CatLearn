""" Script to test the data_setup functions. """
from __future__ import print_function

from ase.ga.data import DataConnection
from atoml.data_setup import get_unique, get_train
from atoml.utilities import remove_outliers

db = DataConnection('gadb.db')

# Get all relaxed candidates from the db file.
all_cand = db.get_all_relaxed_candidates(use_extinct=False)
prune = remove_outliers(all_cand, key='raw_score')
assert len(all_cand) != len(prune)

# Setup the test and training datasets.
testset = get_unique(candidates=all_cand, testsize=10, key='raw_score')
assert len(testset['candidates']) == 10
assert len(testset['taken']) == 10

trainset = get_train(candidates=all_cand, trainsize=50,
                     taken_cand=testset['taken'], key='raw_score')
assert len(trainset['candidates']) == 50
assert len(trainset['target']) == 50
