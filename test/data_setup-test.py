""" Script to test the data_setup functions. """
from ase.ga.data import DataConnection
from predict.data_setup import (get_unique, get_train, remove_outliers,
                                data_split)

db = DataConnection('gadb.db')

# Get all relaxed candidates from the db file.
all_cand = db.get_all_relaxed_candidates(use_extinct=False)
prune = remove_outliers(all_cand, key='raw_score')
assert len(all_cand) != len(prune)

sl = 0
data_split = data_split(all_cand, nsplit=5, key='raw_score')
for i in data_split['split_cand']:
    sl += len(i)
assert sl == len(all_cand)

# Setup the test and training datasets.
testset = get_unique(candidates=all_cand, testsize=10)
assert len(testset['candidates']) == 10
assert len(testset['taken']) == 10

trainset = get_train(candidates=all_cand, trainsize=50,
                     taken_cand=testset['taken'], key='raw_score')
assert len(trainset['candidates']) == 50
assert len(trainset['target']) == 50
