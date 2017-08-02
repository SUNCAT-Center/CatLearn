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

data = np.genfromtxt('fpm.txt')
fpm_train = data[:, :-2]
dbids = data[:, -2]
targets = data[:, -1]


predict_data = np.genfromtxt('fpm_predict.txt')
predict_dbids = predict_data[-1]
fpm_predict = predict_data[:, :-1]

# Set up the kernel.
kdict = {
         # 'lk': {'type': 'linear',
         #       'const': 1.,
         #       'features': [0]},
         'gk': {'type': 'gaussian',
                'width': .3}}

# Get the list of fingerprint vectors and normalize them.
nfp = normalize(train_matrix=fpm_train, test_matrix=fpm_predict)

# Set up the GP.
gp = GaussianProcess(train_fp=nfp['train'], train_target=targets,
                     kernel_dict=kdict,
                     regularization=0.001,
                     optimize_hyperparameters=True)
# Do the prediction
output = gp.get_predictions(test_fp=nfp['test'],
                            get_validation_error=False,
                            get_training_error=False)
y = output['prediction']
uncertainty = output['uncertainty']
predicted_fpm = np.hstack([predict_data, np.vstack(y)])
np.savetxt('prediction.txt', predicted_fpm)
