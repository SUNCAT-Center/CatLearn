# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard
"""

import numpy as np
from atoml.predict import FitnessPrediction

#function to be optimized with respect to theta
def negative_logp(sigma, nfp, targets, regularization):
    """
    Routine to calculate the negative of the  log marginal likelyhood 
    as a function of the hyperparameters, the standardised or 
    normalized fingerprints and the target values.
    
    Input:
        theta: length m+1 vector. Element 0 is lamda for regularization.
            the remaining elements are the m widths (sigma).
        nfp:   n x m matrix
        targets: length n vector
    
    Output:
        -logp, where logp is the the log marginal likelyhood.
    """
    # Set up the prediction routine.
    krr = FitnessPrediction(ktype='gaussian',
        kwidth=sigma,
        regularization=regularization)
    # Get the covariance matrix.
    cov = krr.get_covariance(train_fp=nfp['train'])
    cinv = np.linalg.inv(cov)
    # get the inference level 1 marginal likelyhood.
    logp = krr.log_marginal_likelyhood1(cov,cinv,targets)
    return -logp

