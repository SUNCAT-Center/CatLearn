# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard



"""

import numpy as np
from atoml.fingerprint_setup import normalize
#from adsorbate_fingerprint_mhh import AdsorbateFingerprintGenerator
from atoml.predict import FitnessPrediction

nsplit = 2

#split_cand = []
split_fpv = []
split_energy = []
for i in range(nsplit):
    fpm = np.genfromtxt('fpm_'+str(i)+'.txt')
    split_fpv.append(fpm[:,:-1])
    split_energy.append(fpm[:,-1])

#indexes = [13,0]
indexes = [14,2,1, 9]
#indexes = [15, 2, 1, 9, 11]
#indexes = [1,2,7,12,22,25,26,27,28] #range(39)
#indexes.remove(38)
#indexes.remove(9)
#indexes.remove(18)
#indexes.remove(10)
#indexes.remove(7)


#subset of the fingerprint vector
for i in range(nsplit):
    fpm = split_fpv[i]
    shape = np.shape(fpm)
    reduced_fpv = split_fpv[i][:,indexes]
    split_fpv[i] = reduced_fpv

# Set up the prediction routine.
krr = FitnessPrediction(ktype='gaussian',
                        kwidth=0.5,
                        regularization=0.001)
print('Make predictions based in k-fold samples')
train_rmse = []
val_rmse = []
for i in range(nsplit):
    # Setup the test, training and fingerprint datasets.
    traine = []
    train_fp = []
    testc = []
    teste = []
    test_fp = []
    for j in range(nsplit):
        if i != j:
            for e in split_energy[j]:
                traine.append(e)
            for v in split_fpv[j]:
                train_fp.append(v)
    for e in split_energy[i]:
        teste.append(e)
    for v in split_fpv[i]:
        test_fp.append(v)
    # Get the list of fingerprint vectors and normalize them.
    nfp = normalize(train=train_fp, test=test_fp)
    # Do the training.
    cvm = krr.get_covariance(train_fp=nfp['train'])
    # Do the prediction
    pred = krr.get_predictions(train_fp=nfp['train'],
                               test_fp=nfp['test'],
                               cinv=cvm,
                               train_target=traine,
                               get_validation_error=True,
                               get_training_error=True,
                               test_target=teste)
    # Print the error associated with the predictions.
    train_rmse = pred['training_rmse']['all']
    val_rmse = pred['validation_rmse']['all']
    print('Training error:', np.mean(train_rmse), '+/-', np.std(train_rmse))
    print('Validation error:', np.mean(val_rmse), '+/-', np.std(val_rmse))
