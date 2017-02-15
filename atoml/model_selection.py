# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard
"""
import numpy as np
from .predict import FitnessPrediction


def negative_logp(sigma, nfp, targets, regularization):
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
    krr = FitnessPrediction(ktype='gaussian', kwidth=sigma,
                            regularization=regularization)
    # Get the covariance matrix.
    cov = krr.get_covariance(train_fp=nfp['train'])
    # Invert the covariance matrix.
    cinv = np.linalg.inv(cov)
    # Get the inference level 1 marginal likelyhood.
    logp = krr.log_marginal_likelyhood1(cov=cov, cinv=cinv, y=targets)
    return -logp
