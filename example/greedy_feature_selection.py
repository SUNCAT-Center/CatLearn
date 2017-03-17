# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard



"""

from atoml.fingerprint_setup import normalize, standardize
#from adsorbate_fingerprint_mhh import AdsorbateFingerprintGenerator
from atoml.model_selection import negative_logp
from scipy.optimize import minimize
from atoml.predict import FitnessPrediction
import numpy as np
from matplotlib import pyplot as plt
#import random

nsplit = 2

split_fpv_0 = []
split_energy = []
for i in range(nsplit):
    fpm = np.genfromtxt('fpm_'+str(i)+'.txt')
    split_fpv_0.append(fpm[:,:-2])
    split_energy.append(fpm[:,-2])


split_fpv = list(split_fpv_0)

shape = np.shape(split_fpv_0[0])

#select feature combinations to test
#FEATURES = range(1,shape[1])#forward greedy
start = [53,9,19,36,39,48] #random.randint(0,shape[1])  #forward greedy

FEATURES = start #backward greedy


TRAIN_RMSE = []
VAL_RMSE = []
for fs in FEATURES:
    sigma = None
    indexes = list(FEATURES)    #backward greedy
    indexes.remove(fs)          #backward greedy
    #indexes = start + [fs]        #forward greedy
    print(indexes)
    #subset of the fingerprint vector
    for i in range(nsplit):
        fpm = list(split_fpv_0)[i]
        shape = np.shape(fpm)
        reduced_fpv = split_fpv_0[i][:,indexes]
        split_fpv[i] = reduced_fpv
    #print('Make predictions based in k-fold samples')
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
        # Optimize hyperparameters
        m = np.shape(nfp['train'])[1]
        if sigma == None:
            sigma = np.ones(m)
            sigma *= 0.5
        regularization=.001
        if False:
            a=(nfp, traine, regularization)
            #Hyper parameter bounds.
            b=((1E-9,None),)*(m)
            popt = minimize(negative_logp, sigma, args=a, bounds=b)#, options={'disp': True})
            sigma = popt['x']
        # Set up the prediction routine.
        krr = FitnessPrediction(ktype='gaussian',
                                kwidth=sigma,
                                regularization=regularization)
        cvm = krr.get_covariance(train_matrix=nfp['train'])
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
        train_rmse.append(pred['training_rmse']['average'])
        val_rmse.append(pred['validation_rmse']['average'])
    TRAIN_RMSE.append(train_rmse[0])
    VAL_RMSE.append(np.mean(val_rmse))

plt.scatter(FEATURES, TRAIN_RMSE, c='r')
plt.scatter(FEATURES, VAL_RMSE, c='b')
print('Training error:', min(TRAIN_RMSE), '+/-', np.std(train_rmse))
print('Average Validation error:', min(VAL_RMSE), '+/-', np.std(val_rmse))
plt.show()
