# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve
from .predict import FitnessPrediction
from numpy.core.umath_tests import inner1d

def log_marginal_likelyhood1(K, cinv, y):
    """ Return the log marginal likelyhood.
    (Equation 5.8 in C. E. Rasmussen and C. K. I. Williams, 2006)
    """
    n = len(y)
    y = np.vstack(y)
    L = cholesky(K, lower=True)
    a = cho_solve((L, True), y)
    data_fit = -.5*np.dot(y.T, a)
    complexity = -np.log(np.diag(L)).sum()      #(A.18) in R. & W.
    normalization = -.5*n*np.log(2*np.pi)
    p = (data_fit + complexity + normalization).sum()
    return p

def dkernel_dwidth(fpm_j, width_j, ktype='gaussian'):
    """ Partial derivative of Kernel functions. 
                
        
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
        dkdw_j = np.exp(-.5 * gram**2 / (width_j**2)) * (gram**2 / (width_j**3))
        return  dkdw_j

    # Laplacian kernel.
    elif ktype == 'laplacian':
        raise NotImplementedError('Differentials of Laplacian kernel.')
    
def dK_dwidth_j(train_fp, widths, j):
    """ Returns the partial differential of the covariance matrix with respect
        to the j'th width.
        
        train_fp: list
            A list of the training fingerprint vectors.
            
        widths: float
            A list of the widths or the j'th width.   
            
        j: int
            index of the width, with repsect to which we will differentiate.
    """
    if type(widths) is float:
        width_j = widths
    else:
        width_j = widths[j]
    dK_j = dkernel_dwidth(train_fp[:,j], width_j)
    return dK_j

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
    n=len(y)
    L = cholesky(K, lower=True)
    a = cho_solve((L, True), y)
    C = cho_solve((L, True), np.eye(n))
    aa = np.diag(a**2)
    Q = aa - C
    dp = []
    for j in range(0,m):
        dKdtheta_j = dkernel_dwidth(train_fp[:,j], widths[j])
        # Compute "0.5 * trace(tmp.dot(K_gradient))" without
        # constructing the full matrix tmp.dot(K_gradient) since only
        # its diagonal is required
        dfit = inner1d(a,inner1d(a.T, dKdtheta_j.T))
        dcomp = -np.sum(inner1d(C, dKdtheta_j.T))
        dp.append(0.5*(dfit-dcomp))
    dKdnoise = np.identity(len(train_fp))
    dp.append(0.5*(np.sum(inner1d(Q, dKdnoise.T))))
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
    K = krr.get_covariance(train_fp=train_fp)
    # Invert the covariance matrix.
    cinv = np.linalg.inv(K)
    # Get the inference level 1 marginal likelyhood.
    logp = log_marginal_likelyhood1(K=K, cinv=cinv, y=targets)
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
    w = theta[:-1]
    noise = theta[-1]
    krr = FitnessPrediction(ktype='gaussian', kwidth=w,
                            regularization=noise)
    # Get the covariance matrix.
    K = krr.get_covariance(train_fp=train_fp)
    dlogp = get_dlogp(train_fp, K=K, widths=w, noise=noise, y=targets)
    neg_dlogp = -dlogp
    return neg_dlogp