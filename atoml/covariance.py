# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:27:47 2017

"""

import numpy as np
from .kernels import kernel, kernel_combine, dkernel_dwidth
from atoml import kernels as ak

def general_covariance(train_matrix, kernel_dict={}, regularization=None):
    """ Returns the covariance matrix of training dataset.

        Parameters
        ----------
        train_matrix : list
            A list of the training fingerprint vectors.
    """
    N, N_D = np.shape(train_matrix)
    # Initialize covariance matrix
    cov = np.zeros([N,N])
    # Loop over kernels in kernel_dict
    for key in kernel_dict:
        # Get the type
        ktype = str(kernel_dict[key]['type'])
        
        # Store hyperparameters in single list theta
        if (ktype == 'gaussian' or 
            ktype == 'laplacian') and 'width' in kernel_dict[key]:
            theta = kernel_dict[key]['width']
            if type(theta) is float:
                theta = np.zeros(N_D,) + theta
        
        # Polynomials have pairs of hyperparamters kfree, kdegree
        elif ktype == 'polynomial':
            #kfree = kernel_dict[key]['kfree']
            #kdegree = kernel_dict[key]['kdegree']
            theta = [kernel_dict[key]['kfree'], kernel_dict[key]['kdegree']]
            #if type(kfree) is float:
            #    kfree = np.zeros(N_D,) + kfree
            #if type(kdegree) is float:
            #    kdegree = np.zeros(N_D,) + kdegree
            #zipped_theta = zip(kfree,kdegree)
            # Pass them in order [kfree1, kdegree1, kfree2, kdegree2,...]
            #theta = [hp for k in zipped_theta for hp in k]
        # Linear kernels have no hyperparameters
        elif ktype == 'linear':
            theta = None
        
        # Default hyperparameter keys for other kernels
        elif 'hyperparameters' in kernel_dict[key]:
            theta = kernel_dict[key]['hyperparameters']
            if type(theta) is float:
                theta = np.zeros(N_D,) + theta
        
        elif 'theta' in kernel_dict[key]:
            theta = kernel_dict[key]['theta']
            if type(theta) is float:
                theta = np.zeros(N_D,) + theta

        
        # Select a subset of features for the kernel
        if 'features' in key:
            train_fp = train_matrix[:, kernel_dict[key]['features']]
        else:
            train_fp = train_matrix
                
        # Get the covariance matrix
        if ('operation' in kernel_dict[key] and 
            kernel_dict[key]['operation'] == 'multiplication'):
            cov = cov * eval(
            'ak.'+str(ktype)+'_kernel(train_fp,theta=theta)')
        else:
            cov = cov + eval(
            'ak.'+str(ktype)+'_kernel(train_fp,theta=theta)')
    if regularization is not None:
        cov = cov + regularization * np.identity(len(cov))
    return cov

def testset_covariance(train_matrix, test_matrix, kernel_dict={}):
    """ Returns the covariance matrix of test dataset with training data.

        Parameters
        ----------
        train_matrix : list
            A list of the training fingerprint vectors.
    """
    # Initialize covariance matrix
    N_train, N_D = np.shape(train_matrix)
    N_test, N_D_test = np.shape(test_matrix)
    assert N_D == N_D_test
    cov = np.zeros([N_train, N_test])
    # Loop over kernels in kernel_dict
    for key in kernel_dict:
        # Get the type
        ktype = str(kernel_dict[key]['type'])
        
        # Store hyperparameters in single list theta
        if (ktype == 'gaussian' or 
            ktype == 'laplacian') and 'width' in kernel_dict[key]:
            theta = kernel_dict[key]['width']
            if type(theta) is float:
                theta = np.zeros(N_D,) + theta
        
        # Polynomials have pairs of hyperparamters kfree, kdegree
        elif ktype == 'polynomial':
            #kfree = kernel_dict[key]['kfree']
            #kdegree = kernel_dict[key]['kdegree']
            theta = [kernel_dict[key]['kfree'], kernel_dict[key]['kdegree']]
            #if type(kfree) is float:
            #    kfree = np.zeros(N_D,) + kfree
            #if type(kdegree) is float:
            #    kdegree = np.zeros(N_D,) + kdegree
            #zipped_theta = zip(kfree,kdegree)
            # Pass them in order [kfree1, kdegree1, kfree2, kdegree2,...]
            #theta = [hp for k in zipped_theta for hp in k]
        
        # Linear kernels have no hyperparameters
        elif ktype == 'linear':
            theta = None
        
        # Default hyperparameter keys for other kernels
        elif 'hyperparameters' in kernel_dict[key]:
            theta = kernel_dict[key]['hyperparameters']
            if type(theta) is float:
                theta = np.zeros(N_D,) + theta
        
        elif 'theta' in kernel_dict[key]:
            theta = kernel_dict[key]['theta']
            if type(theta) is float:
                theta = np.zeros(N_D,) + theta
        
        # Select a subset of features for the kernel
        if 'features' in key:
            train_fp = train_matrix[:, kernel_dict[key]['features']]
            test_fp = test_matrix[:, kernel_dict[key]['features']]
        else:
            train_fp = train_matrix
            test_fp = test_matrix
                
        # Get the covariance matrix
        if ('operation' in kernel_dict[key] and 
            kernel_dict[key]['operation'] == 'multiplication'):
            cov = cov * eval(
            'ak.'+str(ktype)+'_kernel(train_fp,test_fp,theta=theta)')
        else:
            cov = cov + eval(
            'ak.'+str(ktype)+'_kernel(train_fp,test_fp,theta=theta)')
    return cov

def get_covariance(train_matrix, ktype='gaussian', 
                   kwidth=None, 
                   kfree=None, 
                   kdegree=None,
                   width_combine=None,
                   combine_kernels=None, 
                   kernel_list=None, 
                   regularization=None):
    """ Returns the covariance matrix between training dataset.

        Parameters
        ----------
        train_matrix : list
            A list of the training fingerprint vectors.
    """
    if type(kwidth) is float:
        kwidth = np.zeros(len(train_matrix[0]),) + kwidth
    if width_combine is None and combine_kernels is not None:
        width_combine = {}
        for k in kernel_list:
            width_combine[k] = kwidth[kernel_list[k]]

    if combine_kernels is None:
        cov = kernel(ktype=ktype, m1=train_matrix, m2=None, 
                     kwidth=kwidth,
                     kfree=kfree,
                     kdegree=kdegree)
    else:
        cov = kernel_combine(combine_kernels=combine_kernels, 
                             kernel_list=kernel_list, 
                             width_combine=width_combine, 
                             m1=train_matrix, m2=None,
                             kwidth=kwidth,
                             kfree=kfree,
                             kdegree=kdegree)

    if regularization is not None:
        cov = cov + regularization * np.identity(len(train_matrix))

    return cov
    
def dK_dwidth_j(train_fp, kwidth, j):
    """ Returns the partial differential of the covariance matrix with respect
        to the j'th width.

        Parameters
        ----------
        train_fp : list
            A list of the training fingerprint vectors.
        widths : float
            A list of the widths or the j'th width.
        j : int
            Index of the width, with repsect to which we will differentiate.
    """
    if type(kwidth) is float:
        width_j = kwidth
    else:
        width_j = kwidth[j]
    dK_j = dkernel_dwidth(train_fp[:, j], width_j)
    return dK_j