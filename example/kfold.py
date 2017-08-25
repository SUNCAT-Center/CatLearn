# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard

This example script requires that the make_fingerprints.py has already been run
or that the user has generated a feature matrix in fpm_(i).txt.
"""
from __future__ import print_function

import numpy as np

from atoml.predict import GaussianProcess
from atoml.feature_preprocess import matrix_split, standardize, normalize
import time

nsplit = 3

fpm_y = np.genfromtxt('fpm.txt')

split_energy = []
split_fpv = []
# Subset of the fingerprint vector.
for i in range(nsplit):
    fpm_y = np.genfromtxt('fpm_' + str(i) + '.txt')
    split_energy.append(fpm_y[:, -1])
    fpm = fpm_y[:, :-2]
    reduced_fpv = fpm
    split_fpv.append(reduced_fpv)
    print(np.shape(reduced_fpv))

print('Make predictions based in k-fold samples')
train_rmse = []
val_rmse = []
start = time.time()
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
    if True:
        # Get the list of fingerprint vectors and standardize them.
        nfp = standardize(train_matrix=train_fp, test_matrix=test_fp)
    else:
        # Get the list of fingerprint vectors and normalize them.
        nfp = normalize(train_matrix=train_fp, test_matrix=test_fp)
    m = np.shape(reduced_fpv)[1]
    sigma = 0.3
    kdict = {
             # 'lk': {'type': 'linear',
             #       'const': 1.,
             #       'features': [0]},
             'gk': {'type': 'sqe',
                    'width': sigma}}  # , 'scaling': 1.}}
    regularization = .003
    # Set up a fresh GP.
    gp = GaussianProcess(train_fp=nfp['train'],
                         train_target=traine,
                         kernel_dict=kdict,
                         regularization=regularization,
                         optimize_hyperparameters=True)
    # Do the training.
    pred = gp.get_predictions(test_fp=nfp['test'],
                              get_validation_error=True,
                              get_training_error=True,
                              test_target=teste)
    # Print the error associated with the predictions.
    train_rmse = pred['training_error']['absolute_average']
    val_rmse = pred['validation_error']['absolute_average']
    print('Training MAE:', train_rmse)
    print('Validation MAE:', val_rmse)
    print(gp.kernel_dict, gp.regularization)
end = time.time()
print(end - start, 'seconds.')