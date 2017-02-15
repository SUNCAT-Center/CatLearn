# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard



"""
from __future__ import print_function

import numpy as np

from atoml.fingerprint_setup import return_fpv, get_combined_descriptors
from atoml.adsorbate_fingerprint import AdsorbateFingerprintGenerator
from atoml.fpm_operations import fpm_operations

fpv_train = AdsorbateFingerprintGenerator(moldb='mol.db',
                                          bulkdb='ref_bulks_k24.db',
                                          slabs='example.db')

# Training and validation data.
train_cand = fpv_train.db2surf_info(selection=['series!=slab'])
print(len(train_cand), 'training candidates.')

train_fpv = [
    fpv_train.primary_addatom,
    fpv_train.primary_adds_nn,
    fpv_train.Z_add,
    fpv_train.adds_sum,
    fpv_train.primary_surfatom,
    fpv_train.primary_surf_nn,
    fpv_train.elemental_dft_properties,
    ]

L_F = get_combined_descriptors(train_fpv)
print(len(L_F), 'descriptors:', L_F)

print('Getting fingerprint vectors.')
cfpv = return_fpv(train_cand, train_fpv)
print('Getting tager values.')
y = return_fpv(train_cand, fpv_train.get_Ef, use_prior=False)
dbid = return_fpv(train_cand, fpv_train.get_dbid, use_prior=False)
print(np.shape(y))
fpm0 = np.hstack([cfpv, np.vstack(y), np.vstack(dbid)])
print('removing any training examples with NaN in the target value field')
fpm = fpm0[~np.isnan(fpm0).any(axis=1)]
print('Saving', np.shape(fpm), 'matrix. Last column is the ase.db id. Second \
      last column is the target value.')
np.savetxt('fpm.txt', fpm)

# fpv_predict = AdsorbateFingerprintGenerator(moldb='mol.db',
#                                            bulkdb='ref_bulks_k24.db',
#                                            slabs='predict.db')
# predict_cand = fpv_predict.db2adds_info('example.db',
#                                        selection=['series!=slab',
#                                                   'id>=50'])
# cfpv_new = return_fpv(predict_cand, train_fpv)
# np.savetxt('fpm_predict.txt', cfpv_new)

nsplit = 2
print('Creating', nsplit, '-fold split.')
ops = fpm_operations(fpm)
split = ops.fpmatrix_split(nsplit)
for i in range(nsplit):
    np.savetxt('fpm_' + str(i) + '.txt', split[i])
