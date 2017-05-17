# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard
"""
from __future__ import print_function

import numpy as np

from atoml.fingerprint_setup import return_fpv, get_combined_descriptors
from atoml.adsorbate_fingerprint import AdsorbateFingerprintGenerator
from atoml.fpm_operations import get_order_2, get_labels_order_2, do_sis, fpmatrix_split

fpv_train = AdsorbateFingerprintGenerator(moldb='mol.db',
                                          bulkdb='ref_bulks_k24.db',
                                          slabs='example.db')

# Training and validation data.
train_cand = fpv_train.db2surf_info(selection=['series!=slab'])
print(len(train_cand), 'training examples.')

train_fpv = [
    #fpv_train.randomfpv,
    #fpv_train.primary_addatom,
    #fpv_train.primary_adds_nn,
    #fpv_train.Z_add,
    #fpv_train.adds_sum,
    fpv_train.primary_surfatom,
    fpv_train.primary_surf_nn,
    fpv_train.elemental_dft_properties,
    ]







L_F = get_combined_descriptors(train_fpv)
print(len(L_F), 'Original descriptors:', L_F)

print('Getting fingerprint vectors.')
cfpv = return_fpv(train_cand, train_fpv)
print('Getting training taget values.')
Ef = return_fpv(train_cand, fpv_train.get_Ef, use_prior=False)
dbid = return_fpv(train_cand, fpv_train.get_dbid, use_prior=False)
print(np.shape(Ef))
assert np.isclose(len(L_F),np.shape(cfpv)[1])
print('Removing any training examples with NaN in the target value field')
fpm0 = np.hstack([cfpv, np.vstack(Ef), np.vstack(dbid)])
fpm_y = fpm0[~np.isnan(fpm0).any(axis=1)]
print('Saving original',np.shape(fpm_y), 'matrix. Last column is the ase.db id. Second last column is the target value.')
np.savetxt('fpm.txt', fpm_y)
fpm = fpm_y[:,:-2]
y = Ef[~np.isnan(fpm0).any(axis=1)]
asedbid=dbid[~np.isnan(fpm0).any(axis=1)]

if True:
    print('Creating combinatorial descriptors.')
    order2 = get_order_2(fpm)
    L_F2 = get_labels_order_2(L_F)
    fpm_c = np.hstack([fpm,order2])
    L = np.hstack([L_F,L_F2])
    assert np.isclose(len(L),np.shape(fpm_c)[1])
    
    print('Perfoming ISIS on '+str(np.shape(fpm_c))+' combinatorial fp matrix.')
    survivors=do_sis(fpm_c, y, size=None, increment=10)
    print(survivors)
    fmd_final=fpm_c[:,survivors]
    l=L[survivors]
    print(len(l), 'surviving descriptors:', l)
    np.savetxt('fpm_sel.txt', fmd_final)
    
    nsplit = 2
    print('Creating', nsplit, '-fold split on survivors.')
    fpm_final_y = np.hstack([fmd_final, np.vstack(y), np.vstack(asedbid)])
    split = fpmatrix_split(fpm_final_y, nsplit)
    for i in range(nsplit):
        np.savetxt('fpm_'+str(i)+'.txt', split[i])
    
    # Making a dataset without taget values. This is not a test set.
    print('Making a new predict-set with the surviving descriptors, without target values.')
    fpv_predict = AdsorbateFingerprintGenerator(moldb='mol.db',
                                                bulkdb='ref_bulks_k24.db',
                                                slabs='predict.db')
    predict_cand = fpv_predict.db2atoms_info('predict.db', selection=['series!=slab'])
    print(len(predict_cand), 'candidates.')
    cfpv_new = return_fpv(predict_cand, train_fpv)
    print('Generating combinatiorial descriptors, and selecting the surviving subset.')
    order2_new = get_order_2(cfpv_new)
    fpm_sel_new = np.hstack([cfpv_new,order2_new])
    fpm_predict = fpm_sel_new[:,survivors]
    np.savetxt('fpm_predict.txt', fpm_predict)


