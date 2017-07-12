# -*- coding: utf-8 -*-
""" This example script optimizes the log marginal likelihood and plots
    how the log marginal likelihoodd varies near the optimum.

    Requirements
    ------------
        data.txt : txt file
            Must contain data where the columns contain descriptors and
            the rows are fingerprints of data points.
"""
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from atoml.feature_preprocess import normalize
from atoml.model_selection import log_marginal_likelihood
import time

# Get the list of fingerprint vectors.
raw_data = np.genfromtxt('data.txt')

# The last column is selected as target value.
targets = raw_data[:, -1]
fingerprints = raw_data[:, :-1]

# Select a few features
features = [1, 2, 3, 4, 5]
fpm_train = fingerprints[:, features]
n, m = np.shape(fpm_train)
print(n, 'training examples', m, 'features')

# Standardize data
nfp = np.array(normalize(train_matrix=fpm_train)['train'])

# Hyperparameter starting guesses.
sigma = np.ones(m)
sigma *= 0.3
regularization = .03
theta = np.append(sigma, regularization)

# Select one or more kernels
kernel_dict = {'k1': {'type': 'gaussian', 'width': list(sigma)}}

# Constant arguments for the log marginal likelihood function
a = (nfp, targets, kernel_dict)

# Hyper parameter bounds.
b = ((1E-9, 1000), ) * (m+1)
print('initial log marginal likelihood =',
      -log_marginal_likelihood(theta,
                               nfp,
                               targets,
                               kernel_dict))

# Optimize hyperparameters
print('Optimizing hyperparameters')
start = time.time()
popt = minimize(log_marginal_likelihood,
                theta, args=a, bounds=b)
end = time.time()
print('Optimized widths = ', popt['x'][:-1])
print('Optimized regularization = ', popt['x'][-1])
print(end - start, 'seconds')

# Plot the log marginal likelihood versus each hyperparameter.
fig, ax = plt.subplots(3, 2)
fname = 'log_marginal_likelihood'
for j in range(m+1):
    axx = j/2
    axy = j % 2
    # Create a space for descriptor j around the optimum.
    thetaspace = np.geomspace(popt['x'][j]/100., popt['x'][j]*100., 65)
    Y = []
    X = []
    theta_copy = popt['x'].copy()
    # Get the log marginal likelihood
    for x in thetaspace:
        X.append(x)
        theta_copy[j] = x
        Y.append(-log_marginal_likelihood(theta_copy,
                                          nfp,
                                          targets,
                                          kernel_dict)
                 )
    # Make the plots.
    ax[axx, axy].plot(X, Y, marker='o')
    # Plot a vertical line at the optimum.
    ax[axx, axy].axvline(popt['x'][j])
    ax[axx, axy].set_xscale('log')
    if j == -1:
        hyperparameter = 'regularization'
    else:
        hyperparameter = 'width_'+str(j)
    ax[axx, axy].set_xlabel(hyperparameter)
    ax[axx, axy].set_ylabel('log marginal likelihood')
# fig.subplots_adjust(hspace=0.5)
fig.tight_layout()
if False:
    fig.savefig(fname+'.pdf', format='pdf')
if True:
    plt.show()
