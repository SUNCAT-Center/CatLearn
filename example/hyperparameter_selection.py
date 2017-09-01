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
from atoml.feature_preprocess import standardize
from atoml.feature_extraction import home_pca
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
sfp = np.array(standardize(train_matrix=fpm_train)['train'])

if False:
    train_matrix = home_pca(5, sfp)['train_fpv']
    n, m = np.shape(train_matrix)
    print(n, 'rows', m, 'principle components')
else:
    train_matrix = sfp

# Select one or more kernels
# kernel_dict = {'k1': {'type': 'gaussian', 'width': [.3] * m}}
kernel_dict = {'k1': {'type': 'sqe', 'width': [.3] * m}}

# Constant arguments for the log marginal likelihood function
a = (train_matrix, targets, kernel_dict)

# Hyperparameter starting guesses.
sigma = np.ones(m) * kernel_dict['k1']['width']
regularization = .003
theta = np.append(sigma, regularization)

# Hyper parameter bounds.
b = ((1E-9, 1e6), ) * (m+1)
print('initial log marginal likelihood =',
      -log_marginal_likelihood(theta,
                               train_matrix,
                               targets,
                               kernel_dict))

# Optimize hyperparameters
print('Optimizing hyperparameters')
start = time.time()
popt = minimize(log_marginal_likelihood,
                theta, args=a, bounds=b)
end = time.time()
print(popt)
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
                                          train_matrix,
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
