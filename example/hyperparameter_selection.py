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
from atoml.model_selection import negative_logp, negative_dlogp, log_marginal_likelihood, gradient_log_p
import time

# Get the list of fingerprint vectors and normalize them.
fpm_raw = np.genfromtxt('fpm.txt')
fpm_train0 = fpm_raw[:, :-1]
fpm_train = fpm_train0[:, :10]  # [6,7,8,10,11,13,20]]
targets = fpm_raw[:, -1]

m = np.shape(fpm_train)[1]
n = len(targets)
nfp = np.array(standardize(train=fpm_train)['train'])

print(n, 'training examples')

# Hyper parameter starting guesses.
sigma = np.ones(m)
sigma *= 0.3
regularization = 0.03
theta = np.append(sigma, regularization)

a = (nfp, targets, 'gaussian', None, None, None)

# Hyper parameter bounds.
b = ((1E-9, None), ) * (m+1)
#print('initial logp=', -negative_logp(theta, nfp, targets))
#print('initial dlogp=', -negative_dlogp(theta, nfp, targets))
print('Optimizing hyperparameters')
start = time.time()
popt = minimize(log_marginal_likelihood, theta,# jac=gradient_log_p,
                                 args=a, bounds=b)
#popt = minimize(negative_logp, theta, args=a, bounds=b, options={'disp': True})
#popt = minimize(negative_logp, theta, args=a, bounds=b, jac=negative_dlogp, options={'disp': True})
#popt = minimize(negative_logp, theta, args=a, jac=negative_dlogp, options={'disp': True}, method='TNC')
end = time.time()
print('Widths aka characteristic lengths = ', popt['x'])
print(end - start, 'seconds')
#print('final logp=', -negative_logp(popt['x'], nfp, targets))
#print('final dlogp=', -negative_dlogp(popt['x'], nfp, targets))
#print(popt)
