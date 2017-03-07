# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard
"""
import numpy as np
from .predict import FitnessPrediction

def log_marginal_likelyhood1(cov, cinv, y):
    """ Return the log marginal likelyhood.
    (Equation 5.8 in C. E. Rasmussen and C. K. I. Williams, 2006)
    """
    n = len(y)
    y = np.vstack(y)
    data_fit = -(np.dot(np.dot(np.transpose(y), cinv), y)/2.)[0][0]
    L = np.linalg.cholesky(cov)
    logdetcov = 0
    for l in range(len(L)):
        logdetcov += np.log(L[l, l])
    complexity = -logdetcov
    normalization = -n*np.log(2*np.pi)/2
    p = data_fit + complexity + normalization
    return p

def dkernel_dsigma(fp1, fp2, sigma, j, ktype='gaussian'):
    """ Derivative of Kernel functions taking two fingerprint vectors. """
    # Linear kernel.
    if ktype == 'linear':
        return 0

    # Polynomial kernel.
    elif ktype == 'polynomial':
        return 0

    # Gaussian kernel.
    elif ktype == 'gaussian':
        return (np.exp(np.abs((fp1 - fp2)[j]) ** 2 / (2 * sigma[j] ** 2)
            ) * (np.abs((fp1 - fp2)[j]) ** 2) / (sigma[j] ** 3) )

    # Laplacian kernel.
    elif ktype == 'laplacian':
        raise NotImplementedError('Differentials of Laplacian kernel.')
    
def dK_dsigma(sigma, train_fp, j):
    """ Returns the covariance matrix between training dataset.

        train_fp: list
            A list of the training fingerprint vectors.
    """
    if type(sigma) is float:
        sigma = np.zeros(len(train_fp[0]),) + sigma

    dK_j = np.asarray([[dkernel_dsigma(fp1, fp2, sigma, j)
                       for fp1 in train_fp] for fp2 in train_fp])
    return dK_j

def negative_logp(theta, nfp, targets):
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
    cov = krr.get_covariance(train_fp=nfp['train'])
    # Invert the covariance matrix.
    cinv = np.linalg.inv(cov)
    # Get the inference level 1 marginal likelyhood.
    logp = log_marginal_likelyhood1(cov=cov, cinv=cinv, y=targets)
    return -logp

def negative_dlogp(theta, nfp, targets):
    """ Routine to calculate the negative of the  log marginal likelyhood as a
        function of the hyperparameters, the standardised or normalized
        fingerprints and the target values.

    Input:
        theta: Length m+1 vector. Element 0 is lamda for regularization.
            The remaining elements are the m widths (sigma).
        nfp: n x m matrix.
        targets: Length n vector.

    Output:
        -dlogp, where dlogp is the the gradient vector of the 
        log marginal likelyhood.
    """
    # Set up the prediction routine.
    sigma = theta[:-1]
    alpha = theta[-1]
    krr = FitnessPrediction(ktype='gaussian', kwidth=sigma,
                            regularization=alpha)
    # Get the covariance matrix.
    cov = krr.get_covariance(train_fp=nfp['train'])
    # Invert the covariance matrix.
    cinv = np.linalg.inv(cov)
    # Get the inference level 1 marginal likelyhood.
    dlogp = krr.dlogp_sckl(cov=cov, cinv=cinv, y=targets)
    return -dlogp