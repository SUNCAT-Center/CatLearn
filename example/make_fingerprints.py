# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard



"""
from __future__ import print_function

import numpy as np

from atoml.fingerprint_setup import (get_fpv, return_fpv,
                                     get_combined_descriptors)
from atoml.adsorbate_fingerprint import AdsorbateFingerprintGenerator

fpv_train = AdsorbateFingerprintGenerator(moldb='mol.db',
                                          bulkdb='ref_bulks_k24.db',
                                          slabs='example.db')

# Training and validation data.
train_cand = fpv_train.db2adds_info('example.db', selection=['series!=slab',
                                                             'id<17'])
print(len(train_cand), 'training candidates')

train_fpv = [
    fpv_train.primary_addatom,
    fpv_train.primary_adds_nn,
    fpv_train.Z_add,
    fpv_train.adds_sum,
    fpv_train.primary_surfatom,
    fpv_train.primary_surf_nn,
    fpv_train.elemental_dft_properties,
    fpv_train.randomfpv
    ]

L_F = get_combined_descriptors(train_fpv)
print(L_F, len(L_F))

print('Getting fingerprint vectors')
cfpv = return_fpv(train_cand, train_fpv)
y = return_fpv(train_cand, fpv_train.get_Ef, use_prior=False)
print(np.shape(y))
fpm = np.hstack([cfpv, np.vstack(y)])
print(np.shape(fpm))
np.savetxt('fpm.txt', fpm)

fpv_predict = AdsorbateFingerprintGenerator(moldb='mol.db',
                                            bulkdb='ref_bulks_k24.db',
                                            slabs='predict.db')
predict_cand = fpv_predict.db2adds_info('example.db',
                                        selection=['series!=slab',
                                                   'id>=17'])
cfpv_new = return_fpv(train_cand, train_fpv)
np.savetxt('fpm_predict.txt', cfpv_new)
