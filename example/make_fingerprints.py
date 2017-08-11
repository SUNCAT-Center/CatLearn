# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard
"""
from __future__ import print_function

import numpy as np

from atoml.fingerprint_setup import return_fpv, get_combined_descriptors
from atoml.db2thermo import (db2mol, db2surf, mol2ref, get_refs, db2surf_info,
                             db2atoms_info, get_formation_energies)
from atoml.adsorbate_fingerprint import AdsorbateFingerprintGenerator
from atoml.utilities import clean_variance, clean_infinite

fname = '../data/example.db'
abinitio_energies, mol_dbids = db2mol('../data/mol.db', ['fmaxout<0.1',
                                                         'pw=500',
                                                         'vacuum=8',
                                                         'psp=gbrv1.5pbe'])
# Prepare atoms objects from database.
e_dict_slabs, id_dict_slabs = db2surf(fname, selection=['series=slab'])
abinitio_energies.update(e_dict_slabs)
print('done slabs')
e_dict, id_dict = db2surf(fname, selection=['series!=slab'])
abinitio_energies.update(e_dict)
mol_dict = mol2ref(abinitio_energies)
ref_dict = get_refs(abinitio_energies, mol_dict)
formation_energies = get_formation_energies(abinitio_energies, ref_dict)

# Training and validation data.
print('Getting atoms objects.')
train_atoms = db2surf_info(fname, id_dict, formation_energies)
print(len(train_atoms), 'Training examples.')

train_gen = AdsorbateFingerprintGenerator()
train_fpv = [
    train_gen.get_dbid,
    # train_gen.randomfpv,
    train_gen.ads_nbonds,
    train_gen.primary_addatom,
    train_gen.primary_adds_nn,
    train_gen.Z_add,
    train_gen.adds_av,
    train_gen.primary_surfatom,
    train_gen.primary_surf_nn,
    ]

L_F = get_combined_descriptors(train_fpv)
print(len(L_F), 'Original descriptors:', L_F)

print('Getting training target values.')
targets = return_fpv(train_atoms, train_gen.info2Ef, use_prior=False)
train_dbid = return_fpv(train_atoms, train_gen.get_dbid, use_prior=False)

print('Getting fingerprint vectors.')
train_raw = return_fpv(train_atoms, train_fpv)
n0_train, d0_train = np.shape(train_raw)
print(n0_train, d0_train)

unlabeled = True
if unlabeled:
    # Making a dataset without taget values. This is not a test set.
    print('Making an unlabeled data set.')
    predict_atoms = db2atoms_info('../data/predict.db',
                                  selection=['series!=slab'])
    predict_raw = return_fpv(predict_atoms, train_fpv)
    n0_predict, d0_predict = np.shape(predict_raw)
    print(n0_predict, d0_predict)
    assert int(d0_predict) == int(d0_train)
else:
    cfpv_predict = None

print('Cleaning up data.')
train_raw = np.array(train_raw, dtype=float)
predict_raw = np.array(predict_raw, dtype=float)
data_dict0 = clean_infinite(train_raw, predict_raw, L_F)
print('Removing features with zero variance.')
data_dict1 = clean_variance(data_dict0['train'], data_dict0['train'],
                            data_dict0['labels'])
n1_train, d1_train = np.shape(data_dict1['train'])
print(d1_train, 'features remain.')
print(np.shape(data_dict1['train']))
train1 = np.hstack([data_dict1['train']])
train1y = np.hstack([data_dict1['train'], np.vstack(targets)])
predict1 = np.hstack([data_dict1['test']])
print('Removing data points with nan values.')
train2y = train1y[np.isfinite(train1y).all(axis=1), :]
predict2 = predict1[np.isfinite(predict1).all(axis=1), :]

print('Saving original', np.shape(train2y), ' training data matrix. ' +
      'Last column is the target value. ' +
      'Second last column is the database id.')
np.savetxt('fpm.txt', train2y)
print('Saving unlabeled', np.shape(predict2), ' data matrix. ' +
      'Last column is the database id.')
np.savetxt('fpm_predict.txt', predict1)
