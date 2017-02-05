# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard



"""
from __future__ import print_function

from sys import argv
from atoml.fingerprint_setup import get_combined_descriptors
from atoml.adsorbate_fingerprint import AdsorbateFingerprintGenerator
import numpy as np

slabs = 'metals.db'

fpv_train = AdsorbateFingerprintGenerator(mols='mol.db',
                                          bulks='ref_bulks_k24.db',
                                          slabs=slabs)

# print('Removing outliers')
# Remove outliers greater than two standard deviations from the median.
# all_cand = remove_outliers(candidates=all_cand, con=1.4826, dev=2.,
#                           key='Ef')

fpv_labels = [
    fpv_train.primary_addatom,
    fpv_train.primary_adds_nn,
    fpv_train.Z_add,
    fpv_train.adds_sum,
    fpv_train.primary_surfatom,
    fpv_train.primary_surf_nn,
    fpv_train.elemental_dft_properties,
    fpv_train.randomfpv
    ]

# print(np.shape(cfpv))

L_F = get_combined_descriptors(fpv_labels)
print(L_F, len(L_F))


try:
    ind = int(argv[1])
    print(L_F[ind])
except ValueError:
    label = argv[1]
    print(label, np.where(L_F == label))

# np.savetxt('fpm.txt', cfpv) #, header=' '.join(L_F))
