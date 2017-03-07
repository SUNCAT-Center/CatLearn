# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard
"""
import numpy as np
from .predict import FitnessPrediction


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
    logp = krr.log_marginal_likelyhood1(cov=cov, cinv=cinv, y=targets)
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