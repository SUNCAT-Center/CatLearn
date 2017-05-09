# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:27:47 2017

"""

import numpy as np
from atoml import kernels as ak

def get_covariance(train_matrix, test_matrix=None, kernel_dict={},
                   regularization=None):
    """ Returns the covariance matrix of training dataset.

        Parameters
        ----------
        train_matrix : list
            A list of the training fingerprint vectors.
    """
    N_train, N_D = np.shape(train_matrix)
    if test_matrix is not None:
        N_test, N_D_test = np.shape(test_matrix)
        assert N_D == N_D_test
    else:
        N_test = N_train
    # Initialize covariance matrix
    cov = np.zeros([N_train,N_test])
    # Loop over kernels in kernel_dict
    for key in kernel_dict:
        ktype = kernel_dict[key]['type']
        
        # Select a subset of features for the kernel
        if 'features' in kernel_dict[key]:
            train_fp = train_matrix[:, kernel_dict[key]['features']]
            if test_matrix is not None:
                test_fp = test_matrix[:, kernel_dict[key]['features']]
            else:
                test_fp = None
            theta = ak.kdict2list(kernel_dict[key], N_D)
        else:
            train_fp = train_matrix
            test_fp = test_matrix
            theta = ak.kdict2list(kernel_dict[key], N_D)
        
        # Get the covariance matrix
        if ('operation' in kernel_dict[key] and 
            kernel_dict[key]['operation'] == 'multiplication'):
            cov = cov * eval(
            'ak.'+str(ktype)+'_kernel(train_fp,test_fp,theta=theta)')
        else:
            cov = cov + eval(
            'ak.'+str(ktype)+'_kernel(train_fp,test_fp,theta=theta)')
    if regularization is not None:
        cov = cov + regularization * np.identity(len(cov))
    return cov

