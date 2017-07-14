# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard
"""
from sys import argv
import numpy as np
from atoml.fingerprint_setup import get_combined_descriptors
from atoml.adsorbate_fingerprint import AdsorbateFingerprintGenerator

train_gen = AdsorbateFingerprintGenerator(bulkdb='ref_bulks_k24.db')
train_fpv = [
    train_gen.get_dbid,
    #train_gen.randomfpv,
    #train_gen.primary_addatom,
    #train_gen.primary_adds_nn,
    #train_gen.Z_add,
    #train_gen.adds_sum,
    train_gen.primary_surfatom,
    train_gen.primary_surf_nn,
    ]

L_F = get_combined_descriptors(train_fpv)

for a in argv[1:]:
    a = a.replace(',','')
    try:
        ind = int(a)
        print(ind, L_F[ind])
    except ValueError:
        label = a
        print(label, np.where(L_F == label)[0][0])