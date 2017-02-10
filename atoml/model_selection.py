# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard
"""

from atoml.predict import FitnessPrediction

#function to be optimized with respect to theta
def negative_logp(theta, nfp, targets):
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
    reg_lambda = theta[0]
    sigma = theta[1:]
    # Set up the prediction routine.
    krr = FitnessPrediction(ktype='gaussian',
        kwidth=sigma,
        regularization=reg_lambda)
    # Get the covariance matrix.
    cinv = krr.get_covariance(train_fp=nfp['train'])
    # get the inference level 1 marginal likelyhood.
    logp = krr.log_marginal_likelyhood1(cinv,targets)
    return -logp
    
def negative_dlogp(theta, nfp, targets):
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
    reg_lambda = theta[0]
    sigma = theta[1:]
    # Set up the prediction routine.
    krr = FitnessPrediction(ktype='gaussian',
        kwidth=sigma,
        regularization=reg_lambda)
    # Get the covariance matrix.
    cinv = krr.get_covariance(train_fp=nfp['train'])
    dK = krr.get_differentials(train_fp=nfp['train'])
    # Get the differential of log the marginal likelyhood
    logp = krr.dlog_marginal_likelyhood1(cinv,dK,targets)
    return -logp

