# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard



"""
from __future__ import print_function

import numpy as np

from atoml.fingerprint_setup import normalize, standardize
from atoml.predict import FitnessPrediction
from atoml.model_selection import negative_logp
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

nsplit = 2 

split_fpv = []
split_energy = []
for i in range(nsplit):
    split = np.genfromtxt('fpm_'+str(i)+'.txt')
    split_fpv.append(split[:, :-1])
    split_energy.append(split[:, -1])

indexes = [12, 6, 7, 1, 15, 18, 4]

# Subset of the fingerprint vector.
for i in range(nsplit):
    fpm = split_fpv[i]
    shape = np.shape(fpm)
    reduced_fpv = split_fpv[i][:, indexes]
    split_fpv[i] = reduced_fpv

print('Make predictions based in k-fold samples')
train_rmse = []
val_rmse = []
colors = ['r','b']
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
    nfp = standardize(train=train_fp, test=test_fp)
    regularization=.001
    m = np.shape(nfp['train'])[1]
    sigma = np.ones(m)
    sigma *= 0.5
    if False:
        # Optimize hyperparameters
        a=(nfp, traine, regularization)
        #Hyper parameter bounds.
        b=((1E-9,None),)*(m)
        popt = minimize(negative_logp, sigma, args=a, bounds=b)
        sigma=popt['x']
    # Set up the prediction routine.
    krr = FitnessPrediction(ktype='gaussian',
                        kwidth=sigma,
                        regularization=.001)#regularization)
    # Do the training.
    cvm = krr.get_covariance(train_fp=nfp['train'])
    cinv = np.linalg.inv(cvm)
    # Do the prediction
    pred = krr.get_predictions(train_fp=nfp['train'],
                               test_fp=nfp['test'],
                               cinv=cinv,
                               train_target=traine,
                               get_validation_error=True,
                               get_training_error=True,
                               test_target=teste)
    # Print the error associated with the predictions.
    train_rmse = pred['training_rmse']['average']
    val_rmse = pred['validation_rmse']['average']
    print('Training error:', train_rmse)
    print('Validation error:', val_rmse)
