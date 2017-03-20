# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard
"""
import numpy as np
import scipy as sp
from scipy.spatial import distance
from .predict import FitnessPrediction
from numpy.core.umath_tests import inner1d

def log_marginal_likelyhood1(cov, cinv, y):
    """ Return the log marginal likelyhood.
    (Equation 5.8 in C. E. Rasmussen and C. K. I. Williams, 2006)
    """
    n = len(y)
    y = np.vstack(y)
    data_fit = -(np.dot(np.dot(np.transpose(y), cinv), y)/2.)
    L = np.linalg.cholesky(cov)
    logdetcov = 0
    for l in range(len(L)):
        logdetcov += np.log(L[l, l])
    complexity = -logdetcov
    normalization = -n*np.log(2*np.pi)/2
    p = data_fit + complexity + normalization
    return p

def dkernel_dsigma(fpm_j, sigma_j, ktype='gaussian'):
    """ Derivative of Kernel functions taking two fingerprint vectors. 
                
        
    """
    # Linear kernel.
    if ktype == 'linear':
        return 0

    # Polynomial kernel.
    elif ktype == 'polynomial':
        return 0

    # Gaussian kernel.
    elif ktype == 'gaussian':
        n=len(fpm_j)
        gram = np.zeros([n,n])
        for i, x1 in enumerate(fpm_j):
            for j, x2 in enumerate(fpm_j):
                if j >= i:
                    break
                d_ij = abs(x1-x2)
                gram[i,j]=d_ij
                gram[j,i]=d_ij
        dk = np.exp(-.5 * gram**2 / (sigma_j**2)) * (gram**2 / (sigma_j**3))
        return  dk

    # Laplacian kernel.
    elif ktype == 'laplacian':
        raise NotImplementedError('Differentials of Laplacian kernel.')
    
def dK_dsigma_j(train_fp, sigma, j):
    """ Returns the partial differential of the covariance matrix with respect
        to the j'th width.
        
        train_fp: list
            A list of the training fingerprint vectors.
            
        sigma: float
            A list of the widths or the j'th width.   
            
        j: int
            index of the width, with repsect to which we will differentiate.
    """
    dK_j = dkernel_dsigma(train_fp[:,j], sigma[j])
    return dK_j

def get_dlogp(train_fp, cov, widths, noise, y):
    """ Return the derived log marginal likelyhood.
    (Equation 5.9 in C. E. Rasmussen and C. K. I. Williams, 2006)
    
    Input:
        train_fp: n x m matrix
        cov: n x n positive definite matrix
        widths: vector of length m
        noise: float
        y: vector of length n
    """
    dimsigma = len(widths)
    L = sp.linalg.cholesky(cov, lower=True)
    a = sp.linalg.cho_solve((L, True), y)
    a2 = np.sum(inner1d(a, a))
    trcinv = sp.linalg.cho_solve((L, True), np.eye(cov.shape[0]))
    dp = []
    for j in range(0,dimsigma):
        dKdtheta_j = dkernel_dsigma(train_fp[:,j], widths[j])
        # Compute "0.5 * trace(tmp.dot(K_gradient))" without
        # constructing the full matrix tmp.dot(K_gradient) since only
        # its diagonal is required
        dpj = 0.5 * np.sum(inner1d(a2-trcinv, dKdtheta_j.T))
        dpj = dpj.sum(-1)
        dp.append(dpj)
    dKdnoise = np.identity(len(train_fp))
    # Compute "0.5 * trace(tmp.dot(K_gradient))" without
    # constructing the full matrix tmp.dot(K_gradient) since only
    # its diagonal is required
    dpj = 0.5 * np.sum(inner1d(a2-trcinv, dKdnoise.T))
    dpj = dpj.sum(-1)
    dp.append(dpj)
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
    sigma = theta[:-1]
    alpha = theta[-1]
    krr = FitnessPrediction(ktype='gaussian', kwidth=sigma,
                            regularization=alpha)
    # Get the covariance matrix.
    cov = krr.get_covariance(train_fp=train_fp)
    # Invert the covariance matrix.
    cinv = np.linalg.inv(cov)
    # Get the inference level 1 marginal likelyhood.
    logp = log_marginal_likelyhood1(cov=cov, cinv=cinv, y=targets)
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
        log marginal likelyhood.
    """
    # Set up the prediction routine.
    sigma = theta[:-1]
    noise = theta[-1]
    krr = FitnessPrediction(ktype='gaussian', kwidth=sigma,
                            regularization=noise)
    # Get the covariance matrix.
    cov = krr.get_covariance(train_fp=train_fp)
    dlogp = get_dlogp(train_fp, cov=cov, widths=sigma, noise=noise, y=targets)
    neg_dlogp = -dlogp
    return neg_dlogp