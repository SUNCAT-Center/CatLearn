# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard
"""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.linalg import cholesky, cho_solve
from numpy.core.umath_tests import inner1d
from .covariance import get_covariance, general_covariance
from .kernels import dkernel_dwidth

def log_marginal_likelihood(theta, train_fp, y, ktype):
    """ Return the log marginal likelyhood.
        (Equation 5.8 in C. E. Rasmussen and C. K. I. Williams, 2006)
    """
    # Get the covariance matrix.
    kdict = {'k1': {'type': ktype, 'theta': theta[:-1]}}
    K = general_covariance(train_fp, kdict, regularization=theta[-1])
    n = len(y)
    y = y.reshape([n, 1])
    # print(np.shape(K), np.max(K), np.min(K))
    L = cholesky(K, lower=True)
    a = cho_solve((L, True), y)
    datafit = -.5*np.dot(y.T, a)
    complexity = -np.log(np.diag(L)).sum()  # (A.18) in R. & W.
    normalization = -.5*n*np.log(2*np.pi)
    p = (datafit + complexity + normalization).sum()
    return -p

def log_marginal_likelyhood1(K, y):
    """ Return the log marginal likelyhood.
        (Equation 5.8 in C. E. Rasmussen and C. K. I. Williams, 2006)
    """
    n = len(y)
    y = y.reshape([n, 1])
    # print(np.shape(K), np.max(K), np.min(K))
    L = cholesky(K, lower=True)
    a = cho_solve((L, True), y)
    datafit = -.5*np.dot(y.T, a)
    complexity = -np.log(np.diag(L)).sum()  # (A.18) in R. & W.
    normalization = -.5*n*np.log(2*np.pi)
    p = (datafit + complexity + normalization).sum()
    return p

def get_dlogp(train_fp, K, widths, noise, y):
    """ Return the gradient of the log marginal likelyhood.
        (Equation 5.9 in C. E. Rasmussen and C. K. I. Williams, 2006)

        Input:
            train_fp: n x m matrix
            K: n x n positive definite matrix
            widths: vector of length m
            noise: float
            y: vector of length n
    """
    m = len(widths)
    n = len(y)
    y = y.reshape([n, 1])
    L = cholesky(K, lower=True)
    a = cho_solve((L, True), y)
    C = cho_solve((L, True), np.eye(n))
    aa = a*a.T  # inner1d(a,a)
    Q = aa - C
    dp = []
    for j in range(0, m):
        dKdtheta_j = dkernel_dwidth(train_fp[:, j], widths[j])
        # Compute "0.5 * trace(tmp.dot(K_gradient))" without
        # constructing the full matrix tmp.dot(K_gradient) since only
        # its diagonal is required
        # dfit = inner1d(aa,dKdtheta_j.T)
        # dcomp = -np.sum(inner1d(C, dKdtheta_j.T))
        # dp.append(0.5*(dfit-dcomp))
        dp.append(0.5*np.sum(inner1d(Q, dKdtheta_j.T)))
    dKdnoise = np.identity(n)
    dp.append(0.5*np.sum(inner1d(Q, dKdnoise.T)))
    return np.array(dp)

def gradient_log_p(theta, train_fp, y, ktype='gaussian', width_combine=None, 
                       combine_kernels=None, kernel_list=None):
    """ Return the gradient of the log marginal likelyhood.
        (Equation 5.9 in C. E. Rasmussen and C. K. I. Williams, 2006)

        Input:
            train_fp: n x m matrix
            K: n x n positive definite matrix
            widths: vector of length m
            noise: float
            y: vector of length n
    """
    if combine_kernels is not None:
        raise NotImplementedError
    if ktype is not 'gaussian':
        raise NotImplementedError
    kwidth = theta[:-1]
    regularization = theta[-1]
    K = get_covariance(train_fp, kwidth=kwidth, width_combine=None, 
                       combine_kernels=None, kernel_list=None, 
                       regularization=regularization)
    m = len(kwidth)
    n = len(y)
    y = y.reshape([n, 1])
    L = cholesky(K, lower=True)
    a = cho_solve((L, True), y)
    C = cho_solve((L, True), np.eye(n))
    aa = a*a.T  # inner1d(a,a)
    Q = aa - C
    dp = []
    for j in range(0, m):
        dKdtheta_j = dkernel_dwidth(train_fp[:, j], kwidth[j])
        # Compute "0.5 * trace(tmp.dot(K_gradient))" without
        # constructing the full matrix tmp.dot(K_gradient) since only
        # its diagonal is required
        # dfit = inner1d(aa,dKdtheta_j.T)
        # dcomp = -np.sum(inner1d(C, dKdtheta_j.T))
        # dp.append(0.5*(dfit-dcomp))
        dp.append(0.5*np.sum(inner1d(Q, dKdtheta_j.T)))
    dKdnoise = np.identity(n)
    dp.append(0.5*np.sum(inner1d(Q, dKdnoise.T)))
    return np.array(dp)

def negative_logp(theta, train_fp, targets):
    """ Routine to calculate the negative of the  log marginal likelyhood as a
        function of the hyperparameters, the standardised or normalized
        fingerprints and the target values.

        Input:
            theta: Length m+1 vector. Element 0 is lamda for regularization.
                The remaining elements are the m widths (sigma).
            nfp: n x m matrix.
            targets: Length n vector.

        Output:
            -logp, where logp is the the log marginal likelyhood.
    """
    # Set up the prediction routine.
    kwidth = theta[:-1]
    regularization = theta[-1]
    # Get the covariance matrix.
    K = get_covariance(train_fp, kwidth=kwidth, width_combine=None, 
                       combine_kernels=None, kernel_list=None, 
                       regularization=regularization)
    # Get the inference level 1 marginal likelyhood.
    logp = log_marginal_likelyhood1(K=K, y=targets)
    return -logp


def negative_dlogp(theta, train_fp, targets):
    """ Routine to calculate the negative of the derived log marginal
        likelyhood as a function of the hyperparameters,
        the standardised or normalized fingerprints and the target values.

        Input:
            theta: Length m+1 vector. Element 0 is the noise parameter.
                The remaining elements are the m widths (sigma).
            nfp: n x m matrix.
            targets: Length n vector.

        Output:
            -dlogp, where dlogp is the the gradient vector of the
            log marginal likelyhood with respect to the hyperparameters.
    """
    # Set up the prediction routine.
    kwidth = theta[:-1]
    regularization = theta[-1]
    # Get the covariance matrix.
    K = get_covariance(train_fp, kwidth=kwidth, width_combine=None, 
                       combine_kernels=None, kernel_list=None, 
                       regularization=regularization)
    dlogp = get_dlogp(train_fp, K=K, widths=kwidth, 
                      noise=regularization, y=targets)
    neg_dlogp = -dlogp
    return neg_dlogp
