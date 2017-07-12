# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard

This example script requires that the make_fingerprints.py has already been run
or that the user has generated a feature matrix in fpm.txt.
"""
from __future__ import print_function

import numpy as np

from atoml.feature_preprocess import normalize
# from adsorbate_fingerprint_mhh import AdsorbateFingerprintGenerator
from atoml.predict import GaussianProcess

fpm_raw = np.genfromtxt('fpm.txt')
fpm_train0 = fpm_raw[:, :-2]
targets = fpm_raw[:, -2]

fpm_predict0 = np.genfromtxt('fpm_predict.txt')[:-2]

indexes = [6, 7, 16, 11]  # feature indexes

fpm_train = fpm_train0[:, indexes]
fpm_predict = fpm_predict0[:, indexes]

# Set up the kernel.
kdict = {
         # 'lk': {'type': 'linear',
         #       'const': 1.,
         #       'features': [0]},
         'gk': {'type': 'gaussian',
                'width': .3}}

# Set up the prediction routine.
gp = GaussianProcess(kernel_dict=kdict,
                     regularization=0.001)
# Get the list of fingerprint vectors and normalize them.
nfp = normalize(train_matrix=fpm_train, test_matrix=fpm_predict)
# Do the prediction
output = gp.get_predictions(train_fp=nfp['train'],
                             test_fp=nfp['test'],
                             train_target=targets,
                             get_validation_error=False,
                             get_training_error=False)
y = output['prediction']

predicted_fpm = np.hstack([fpm_predict0, np.vstack(y)])
np.savetxt('prediction.txt', predicted_fpm)
