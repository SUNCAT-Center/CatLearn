# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:52:25 2017

Contains kernel functions and gradients of kernels.

"""
import numpy as np
from scipy.spatial import distance

def gaussian_kernel(m1, m2=None, theta=None):
    kwidth = theta
    if m2 is None:
        k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)
        return k
    else:
        k = distance.cdist(m1 / kwidth, m2 / kwidth,
                           metric='sqeuclidean')
        return np.exp(-.5 * k)

def d_gaussian_kernel(m1, m2=None, theta=None):
    kwidth = theta
    n = len(fpm_j)
    gram = np.zeros([n, n])
    # Construct Gram matrix.
    for i, x1 in enumerate(fpm_j):
        for j, x2 in enumerate(fpm_j):
            if j >= i:
                break
            d_ij = abs(x1-x2)
            gram[i, j] = d_ij
            gram[j, i] = d_ij
    # Insert gram matrix in differentiated kernel.
    dkdw_j = np.exp(-.5 * gram**2 / (width_j**2)) * (gram**2 /
                                                     (width_j**3))
    return dkdw_j

def linear_kernel(m1, m2=None, theta=None):
    if m2 is None:
        m2 = m1
    return np.dot(m1, np.transpose(m2))

def polynomial_kernel(m1, m2=None, theta=None):
    kfree = theta[0]
    kdegree = theta[1]
    if m2 is None:
        m2 = m1
    return(np.dot(m1, np.transpose(m2)) + kfree) ** kdegree

def d_polynomial_kernel(m1, m2=None, theta=None):
    return(np.dot(m1, np.transpose(m2)) + kfree) ** kdegree-1
    raise NotImplementedError('To Do')

def laplacian_kernel(m1, m2=None, theta=None):
    kwidth = theta
    if m2 is None:
        k = distance.pdist(m1 / kwidth, metric='cityblock')
        k = distance.squareform(np.exp(-k))
        np.fill_diagonal(k, 1)
        return k
    else:
        k = distance.cdist(m1 / kwidth, m2 / kwidth,
                               metric='cityblock')
        return np.exp(-k)

    