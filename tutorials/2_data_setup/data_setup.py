"""Script to show data generation functions."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from random import random

from ase.ga.data import DataConnection

from atoml.utilities.data_setup import get_unique, get_train
from atoml.fingerprint.setup import return_fpv
from atoml.fingerprint import StandardFingerprintGenerator
from atoml.utilities import DescriptorDatabase

# Connect ase atoms database.
gadb = DataConnection('../../data/gadb.db')

# Get all relaxed candidates from the db file.
print('Getting candidates from the database')
all_cand = gadb.get_all_relaxed_candidates(use_extinct=False)

# Setup the test and training datasets.
testset = get_unique(atoms=all_cand, size=10, key='raw_score')

trainset = get_train(atoms=all_cand, size=50, taken=testset['taken'],
                     key='raw_score')

# Clear out some old saved data.
for i in trainset['atoms']:
    del i.info['data']['nnmat']

# Initiate the fingerprint generators with relevant input variables.
print('Getting the fingerprints')
sfpv = StandardFingerprintGenerator(atom_types=[78, 79])

data = return_fpv(trainset['atoms'], [sfpv.eigenspectrum_fpv],
                  use_prior=False)

# Define variables for database to store system descriptors.
db_name = 'fpv_store.sqlite'
descriptors = ['f' + str(i) for i in range(np.shape(data)[1])]
targets = ['Energy']
names = descriptors + targets

# Set up the database to save system descriptors.
dd = DescriptorDatabase(db_name=db_name, table='FingerVector')
dd.create_db(names=names)

# Put data in correct format to be inserted into database.
print('Generate the database')
new_data = []
for i, a in zip(data, all_cand):
    d = []
    d.append(a.info['unique_id'])
    for j in i:
        d.append(j)
    d.append(a.info['key_value_pairs']['raw_score'])
    new_data.append(d)

# Fill the database with the data.
dd.fill_db(descriptor_names=names, data=new_data)

# Test out the database functions.
train_fingerprint = dd.query_db(names=descriptors)
train_target = dd.query_db(names=targets)
print('\nfeature data for candidates:\n', train_fingerprint,
      '\ntarget data for candidates:\n', train_target)

all_id = dd.query_db(names=['uuid'])
dd.create_column(new_column=['random'])
for i in all_id:
    dd.update_descriptor(descriptor='random', new_data=random(),
                         unique_id=i[0])
print('\nretrieve random vars:\n', dd.query_db(names=['random']))

print('\nretrieved column names:\n', dd.get_column_names())
