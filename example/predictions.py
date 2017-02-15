# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard



"""
from __future__ import print_function

import numpy as np

from atoml.fingerprint_setup import normalize
# from adsorbate_fingerprint_mhh import AdsorbateFingerprintGenerator
from atoml.predict import FitnessPrediction

fpm_raw = np.genfromtxt('fpm.txt')
fpm_train0 = fpm_raw[:, :-1]
targets = fpm_raw[:, -1]

fpm_predict0 = np.genfromtxt('fpm_predict.txt')

indexes = [6, 7, 16, 11]  # feature indexes

fpm_train = fpm_train0[:, indexes]
fpm_predict = fpm_predict0[:, indexes]

# Set up the prediction routine.
krr = FitnessPrediction(ktype='gaussian',
                        kwidth=0.5,
                        regularization=0.001)
# Get the list of fingerprint vectors and normalize them.
nfp = normalize(train=fpm_train, test=fpm_predict)
# Do the training.
cvm = krr.get_covariance(train_fp=nfp['train'])
cinv = np.linalg.inv(cvm)
# Do the prediction
output = krr.get_predictions(train_fp=nfp['train'],
                             test_fp=nfp['test'],
                             cinv=cvm,
                             train_target=targets,
                             get_validation_error=False,
                             get_training_error=False)
y = output['prediction']

predicted_fpm = np.hstack([fpm_predict0, np.vstack(y)])
np.savetxt('prediction.txt', predicted_fpm)
