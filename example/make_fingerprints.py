# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard



"""
from __future__ import print_function

from atoml.fingerprint_setup import (get_single_fpv, get_combined_fpv,
                                     get_combined_descriptors)
from atoml.adsorbate_fingerprint import AdsorbateFingerprintGenerator
import numpy as np

fpv_train = AdsorbateFingerprintGenerator(mols='mol.db',
                                          bulks='ref_bulks_k24.db',
                                          slabs='example.db')
# fpv_new = AdsorbateFingerprintGenerator(mols='mol.db',
#                                        bulks='ref_bulks_k24.db')

# training and validation data
train_cand = fpv_train.db2surf_info()  # fname='metals.db',
#                                    selection=['Al=0', 'series=H', 'layers=5',
#                                               'facet=1x1x1', 'kpts=4x6',
#                                               'PW=500', 'PSP=gbrv1.5pbe'])
print(len(train_cand), 'training candidates')

# test data (no Ef)
# new_cand = fpv_new.db2atoms_info(fname='input_mslab.db',
#                                 selection=['id>400', 'Al=0', 'series=H',
#                                            'layers=5', 'facet=1x1x1',
#                                            'supercell=3x2'])
# print(len(new_cand), 'new candidates')


# print('Removing outliers')
# Remove outliers greater than two standard deviations from the median.
# all_cand = remove_outliers(candidates=all_cand, con=1.4826, dev=2.,
#                           key='Ef')

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

# new_fpv = [
#    fpv_new.primary_addatom,
#    fpv_new.primary_adds_nn,
#    fpv_new.Z_add,
#    fpv_new.adds_sum,
#    fpv_new.primary_surfatom,
#    fpv_new.primary_surf_nn,
#    fpv_new.elemental_dft_properties,
#    fpv_new.randomfpv
#    ]

L_F = get_combined_descriptors(train_fpv)
print(L_F, len(L_F))

# new_L_F = get_combined_descriptors(new_fpv)
# print(new_L_F, len(new_L_F))

# assert np.all( L_F == new_L_F )

print('Getting fingerprint vectors')
# new_fpm = get_combined_fpv(new_cand, new_fpv)
# assert len(L_F) == np.shape(new_fpm)[1]
# print(np.shape(new_fpm))
# np.savetxt('fpm_predict.txt', new_fpm) #, header=' '.join(L_F))

cfpv = get_combined_fpv(train_cand, train_fpv)
y = get_single_fpv(train_cand, fpv_train.get_Ef, use_prior=False)
print(np.shape(y))
fpm = np.hstack([cfpv, np.vstack(y)])
print(np.shape(fpm))
np.savetxt('fpm.txt', fpm)  # , header=' '.join(L_F
