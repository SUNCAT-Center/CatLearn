# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard

This example script requires that the make_fingerprints.py has already been run
or that the user has generated a feature matrix in fpm_(i).txt.
"""
from __future__ import print_function

import numpy as np

from atoml.fingerprint_setup import standardize, normalize
from atoml.predict import GaussianProcess
from atoml.fpm_operations import fpmatrix_split

nsplit = 2

fpm_y = np.genfromtxt('fpm.txt')
split = fpmatrix_split(fpm_y, nsplit)
indexes = [9, 18, 25, 32, 36]

split_energy = []
split_fpv = []
# Subset of the fingerprint vector.
for i in range(nsplit):
    split_energy.append(split[i][:, -2])
    fpm = split[i][:, :-2]
    reduced_fpv = fpm[:, indexes]
    split_fpv.append(reduced_fpv)
    print(np.shape(reduced_fpv))
    print(np.shape(split_energy[i]))

print('Make predictions based in k-fold samples')
train_rmse = []
val_rmse = []
sigma = None
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
    regularization = .01
    m = np.shape(reduced_fpv)[1]
    if sigma is None:
        sigma = np.ones(m)
        sigma *= 0.3
        kdict = {'kernel': {'type': 'gaussian', 'width': list(sigma)}}
    if True:
        # Get the list of fingerprint vectors and standardize them.
        nfp = standardize(train=train_fp, test=test_fp)
    else:
        # Get the list of fingerprint vectors and normalize them.
        nfp = normalize(train=train_fp, test=test_fp)
    # Set up the prediction routine.
    krr = GaussianProcess(kernel_dict=kdict,
                            regularization=regularization)  # regularization)
    # Do the training.
    pred = krr.get_predictions(train_fp=nfp['train'],
                               test_fp=nfp['test'],
                               cinv=None,
                               train_target=traine,
                               get_validation_error=True,
                               get_training_error=True,
                               test_target=teste, 
                               optimize_hyperparameters=True)
    # Print the error associated with the predictions.
    train_rmse = pred['training_rmse']['average']
    val_rmse = pred['validation_rmse']['average']
    print('Training error:', train_rmse)
    print('Validation error:', val_rmse)
