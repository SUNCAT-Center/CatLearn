# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:52:25 2017

Contains kernel functions and gradients of kernels.

"""
import numpy as np
from scipy.spatial import distance

def kdict2list(kdict, N_D=None):
    """
        Returns an ordered list of hyperparameters, given a dictionary containing
        properties of a single kernel. The dictionary must contain either the key
        'hyperparameters' or 'theta' containing a list of hyperparameters 
        or the keys 'type' containing the type name in a string and 
        'width' in the case of a 'gaussian' or 'laplacian' type or the keys
        'kdegree' and 'kfree' in the case of a 'polynomial' type.
            
        Parameters
        ----------
        kdict : dict
            A kernel dictionary containing the keys 'type' and optional
            keys containing the hyperparameters of the kernel.
        
        N_D : none or int
    
    """
    # Get the type
    ktype = str(kdict['type'])
    
    # Store hyperparameters in single list theta
    if (ktype == 'gaussian' or 
        ktype == 'laplacian') and 'width' in kdict:
        theta = kdict['width']
        if 'features' in kdict:
            N_D = len(kdict['features'])
        elif N_D is None:
            N_D = len(kdict['width'])
        if type(theta) is float:
            theta = np.zeros(N_D,) + theta
    
    # Polynomials have pairs of hyperparamters kfree, kdegree
    elif ktype == 'polynomial':
        #kfree = kernel_dict[key]['kfree']
        #kdegree = kernel_dict[key]['kdegree']
        theta = [kdict['kfree'], kdict['kdegree']]
        #if type(kfree) is float:
        #    kfree = np.zeros(N_D,) + kfree
        #if type(kdegree) is float:
        #    kdegree = np.zeros(N_D,) + kdegree
        #zipped_theta = zip(kfree,kdegree)
        # Pass them in order [kfree1, kdegree1, kfree2, kdegree2,...]
        #theta = [hp for k in zipped_theta for hp in k]
    # Linear kernels have no hyperparameters
    elif ktype == 'linear':
        theta = []
    
    # Default hyperparameter keys for other kernels
    elif 'hyperparameters' in kdict:
        theta = kdict['hyperparameters']
        if 'features' in kdict:
            N_D = len(kdict['features'])
        elif N_D is None:
            N_D = len(theta)
        if type(theta) is float:
            theta = np.zeros(N_D,) + theta
    
    elif 'theta' in kdict:
        theta = kdict['theta']
        if 'features' in kdict:
            N_D = len(kdict['features'])
        elif N_D is None:
            N_D = len(theta)
        if type(theta) is float:
            theta = np.zeros(N_D,) + theta
    return theta

def kdicts2list(kernel_dict, N_D=None):
    """
        Returns an ordered list of hyperparameters given the kernel dictionary.
        The kernel dictionary must contain one or more dictionaries, each 
        specifying the type and hyperparameters.
        
        Parameters
        ----------
        kernel_dict : dict
            A dictionary containing kernel dictionaries.
        
        N_D : none or int
    """
    theta=[]
    for kernel_key in kernel_dict:
        theta.append(kdict2list(kernel_dict[kernel_key], N_D=N_D))
    hyperparameters = np.concatenate(theta)
    return hyperparameters

def list2kdict(hyperparameters, kernel_dict):
    """
        Returns a updated kernel dictionary with updated hyperparameters, 
        given an ordered list of hyperparametersthe and the previous kernel 
        dictionary. The kernel dictionary must contain a dictionary for each 
        kernel type in the same order as their respective hyperparameters 
        in the list hyperparameters.
        
        Parameters
        ----------
        hyperparameters : list
        
        kernel_dict : dict
            A dictionary containing kernel dictionaries.
    
    """
    ki=0
    for key in kernel_dict:
        ktype = kernel_dict[key]['type']
        # Retreive hyperparameters from a single list theta
        if (ktype == 'gaussian' or ktype == 'laplacian'):
            N_D = len(kernel_dict[key]['width'])
            theta = hyperparameters[ki:ki+N_D]
            kernel_dict[key]['width'] = theta
            ki += N_D
        
        # Polynomials have pairs of hyperparamters kfree, kdegree
        elif ktype == 'polynomial':
            theta = hyperparameters[ki:ki+2]
            kernel_dict[key]['kfree'] = theta[0]
            kernel_dict[key]['kdegree'] = theta[1]
            ki += 2
    
        # Linear kernels have no hyperparameters
        elif ktype == 'linear':
            continue
        
        # Default hyperparameter keys for other kernels
        else:
            N_D = len(kernel_dict[key]['hyperparameters'])
            theta = hyperparameters[ki:ki+N_D]
            kernel_dict[key]['hyperparameters'] = theta
    return kernel_dict

def gaussian_kernel(m1, m2=None, theta=None):
    """
        Returns the covariance matrix between datasets m1 and m2 
        with a gaussian kernel.
        
        Parameters
        ----------
        m1 : list
            A list of the training fingerprint vectors.

        m2 : list or None
            A list of the training fingerprint vectors.
            
        theta : list
        
    """
    kwidth = theta
    if m2 is None:
        k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)
        return k
    else:
        k = distance.cdist(m1 / kwidth, m2 / kwidth,
                           metric='sqeuclidean')
        return np.exp(-.5 * k)

def d_gaussian_kernel(fpm_j, theta=None):
    """
        W.I.P.
    """
    kwidth_j = theta
    n = len(fpm_j)
    gram = np.zeros([n, n])
    # Construct Gram matrix.
    for i, x1 in enumerate(fpm_j):
        for j, x2 in enumerate(fpm_j):
            if j >= i:
                break
            d_ij = abs(x1-x2)
            gram[i, j] = d_ij
            gram[j, i] = d_ij
    # Insert gram matrix in differentiated kernel.
    dkdw_j = np.exp(-.5 * gram**2 / (kwidth_j**2)) * (gram**2 /
                                                     (kwidth_j**3))
    return dkdw_j

def linear_kernel(m1, m2=None, theta=None):
    """
        Returns the covariance matrix between datasets m1 and m2 
        with a linear kernel.
        
        Parameters
        ----------
        m1 : list
            A list of the training fingerprint vectors.

        m2 : list or None
            A list of the training fingerprint vectors.
            
        theta : list
        
    """
    if m2 is None:
        m2 = m1
    return np.dot(m1, np.transpose(m2))

def polynomial_kernel(m1, m2=None, theta=None):
    """
        Returns the covariance matrix between datasets m1 and m2 
        with a polynomial kernel.
        
        Parameters
        ----------
        m1 : list
            A list of the training fingerprint vectors.

        m2 : list or None
            A list of the training fingerprint vectors.
            
        theta : list
        
    """
    kfree = theta[0]
    kdegree = theta[1]
    if m2 is None:
        m2 = m1
    return(np.dot(m1, np.transpose(m2)) + kfree) ** kdegree

def d_polynomial_kernel(m1, m2=None, theta=None):
    """
        W.I.P.
    """
    raise NotImplementedError('To Do')

def laplacian_kernel(m1, m2=None, theta=None):
    kwidth = theta
    if m2 is None:
        k = distance.pdist(m1 / kwidth, metric='cityblock')
        k = distance.squareform(np.exp(-k))
        np.fill_diagonal(k, 1)
        return k
    else:
        k = distance.cdist(m1 / kwidth, m2 / kwidth,
                               metric='cityblock')
        return np.exp(-k)

def dkernel_dwidth(fpm_j, width_j, ktype='gaussian'):
    """ 
        Returns the partial derivative of kernel functions. 
        
        W.I.P.
    """
    # Linear kernel.
    if ktype == 'linear':
        return 0

    # Polynomial kernel.
    elif ktype == 'polynomial':
        raise NotImplementedError('Differentials of polynomial kernel.')

    # Gaussian kernel.
    elif ktype == 'gaussian':
        n = len(fpm_j)
        gram = np.zeros([n, n])
        # Construct Gram matrix.
        for i, x1 in enumerate(fpm_j):
            for j, x2 in enumerate(fpm_j):
                if j >= i:
                    break
                d_ij = abs(x1-x2)
                gram[i, j] = d_ij
                gram[j, i] = d_ij
        # Insert gram matrix in differentiated kernel.
        dkdw_j = np.exp(-.5 * gram**2 / (width_j**2)) * (gram**2 /
                                                         (width_j**3))
        return dkdw_j

    # Laplacian kernel.
    elif ktype == 'laplacian':
        raise NotImplementedError('Differentials of Laplacian kernel.')
    