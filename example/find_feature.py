# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard
"""
from __future__ import print_function

from atoml.fingerprint_setup import get_combined_descriptors
from atoml.adsorbate_fingerprint import AdsorbateFingerprintGenerator

slabs = 'example.db'

fpv_train = AdsorbateFingerprintGenerator(moldb='mol.db',
                                          bulkdb='ref_bulks_k24.db',
                                          slabs=slabs)

fpv_labels = [
    fpv_train.primary_addatom,
    fpv_train.primary_adds_nn,
    fpv_train.Z_add,
    fpv_train.adds_sum,
    fpv_train.primary_surfatom,
    fpv_train.primary_surf_nn,
    fpv_train.elemental_dft_properties,
    ]

L_F = get_combined_descriptors(fpv_labels)
print(L_F, len(L_F))
