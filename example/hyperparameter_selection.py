# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard

This example script requires that the make_fingerprints.py has already been run
or that the user has generated a feature matrix in fpm.txt.
"""
import numpy as np
from scipy.optimize import minimize

from atoml.fingerprint_setup import standardize
from atoml.model_selection import negative_logp

# Get the list of fingerprint vectors and normalize them.
fpm_raw = np.genfromtxt('fpm.txt')
fpm_train0 = fpm_raw[:, :-1]
fpm_train = fpm_train0[:, :21]  # [6,7,8,10,11,13,20]]
targets = fpm_raw[:, -1]

m = np.shape(fpm_train)[1]
n = len(targets)
nfp = standardize(train=fpm_train)

# Hyper parameter starting guesses.
sigma = np.ones(m)
sigma *= 0.5
regularization = 0.1
theta = np.append(sigma, regularization)

a = (nfp, targets)

# Hyper parameter bounds.
b = ((1E-9, None), ) * (m+1)
print('Optimizing hyperparameters')
popt = minimize(negative_logp, theta, args=a, bounds=b)
print('Widths aka characteristic lengths = ', popt['x'])
p = -negative_logp(popt['x'], nfp, targets)
print(popt)
