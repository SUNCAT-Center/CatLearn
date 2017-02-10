""" Script to test the data_setup functions. """
from __future__ import print_function

from ase.ga.data import DataConnection
from atoml.data_setup import get_unique, get_train, remove_outliers, data_split

db = DataConnection('gadb.db')

# Get all relaxed candidates from the db file.
all_cand = db.get_all_relaxed_candidates(use_extinct=False)
prune = remove_outliers(all_cand, key='raw_score')
assert len(all_cand) != len(prune)

sl = 0
split_all = data_split(all_cand, nsplit=5, key='raw_score')
for i in split_all['split_cand']:
    sl += len(i)
assert sl == len(all_cand)

split_fixed = data_split(all_cand, nsplit=5, key='raw_score', fix_size=100,
                         replacement=True)
for i in split_fixed['split_cand']:
    assert len(i) == 100

# Setup the test and training datasets.
testset = get_unique(candidates=all_cand, testsize=10, key='raw_score')
assert len(testset['candidates']) == 10
assert len(testset['taken']) == 10

trainset = get_train(candidates=all_cand, trainsize=50,
                     taken_cand=testset['taken'], key='raw_score')
assert len(trainset['candidates']) == 50
assert len(trainset['target']) == 50
