# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:52:25 2017

Contains kernel functions and gradients of kernels.

"""
import numpy as np
from scipy.spatial import distance

def gaussian_kernel(m1, m2=None, theta=None):
    kwidth = theta
    if m2 is None:
        k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)
    else:
        k = distance.cdist(m1 / kwidth, m2 / kwidth,
                           metric='sqeuclidean')
    return np.exp(-.5 * k)

def d_gaussian_kernel(m1, m2=None, theta=None):
    kwidth = theta
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

def linear_kernel(m1, m2=None, theta=None):
    if m2 is None:
        m2 = m1
    return np.dot(m1, np.transpose(m2))

def d_linear_kernel(m1, m2=None, theta=None):
    return np.dot(m1, np.transpose(m2))

def polynomial_kernel(m1, m2=None, theta=None):
    kfree = theta[::2]
    kdegree = theta[1::2]
    if m2 is None:
        m2 = m1
    return(np.dot(m1, np.transpose(m2)) + kfree) ** kdegree

def d_polynomial_kernel(m1, m2=None, theta=None):
    raise NotImplementedError('To Do')

def laplacian_kernel(m1, m2, theta=None):
    if m2 is None:
        kwidth = theta
        k = distance.pdist(m1 / kwidth, metric='cityblock')
        k = distance.squareform(np.exp(-k))
        np.fill_diagonal(k, 1)
    else:
        k = distance.cdist(m1 / kwidth, m2 / kwidth,
                               metric='cityblock')
    return np.exp(-k)

def kernel(ktype, m1, m2=None, kwidth=None, kfree=None, kdegree=None):
    """ Kernel functions taking n x d feature matrix.

        Parameters
        ----------
        m1 : array
            Feature matrix for training (or test) data.
        m2 : array
            Feature matrix for test data.

        Returns
        -------
        Kernelized representation of the feature space as array.
    """
    if m2 is None:
        # Gaussian kernel.
        if ktype is 'gaussian':
            k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
            k = distance.squareform(np.exp(-.5 * k))
            np.fill_diagonal(k, 1)
            return k

        # Laplacian kernel.
        elif ktype is 'laplacian':
            k = distance.pdist(m1 / kwidth, metric='cityblock')
            k = distance.squareform(np.exp(-k))
            np.fill_diagonal(k, 1)
            return k

        # Otherwise set m2 equal to m1 as functions are the same.
        m2 = m1

    # Linear kernel.
    if ktype is 'linear':
        return np.dot(m1, np.transpose(m2))

    # Polynomial kernel.
    elif ktype is 'polynomial':
        return(np.dot(m1, np.transpose(m2)) + kfree) ** kdegree

    # Gaussian kernel.
    elif ktype is 'gaussian':
        k = distance.cdist(m1 / kwidth, m2 / kwidth,
                           metric='sqeuclidean')
        return np.exp(-.5 * k)

    # Laplacian kernel.
    elif ktype is 'laplacian':
        k = distance.cdist(m1 / kwidth, m2 / kwidth,
                           metric='cityblock')
        return np.exp(-k)

def kernel_combine(combine_kernels, kernel_list, width_combine, m1, m2=None, 
                   kwidth=None, kfree=None, kdegree=None):
    """ Function to generate a covarience matric with a combination of
        kernel functions.

        Parameters
        ----------
        m1 : array
            Feature matrix for training (or test) data.
        m2 : array
            Feature matrix for test data.

        Returns
        -------
        Combined kernelized representation of the feature space as array.
    """
    msg = 'Must combine covarience from more than one kernel.'
    assert len(kernel_list) > 1, msg
    
    # Form additive covariance matrix.
    if combine_kernels is 'addition':
        if m2 is None:
            c = np.zeros((np.shape(m1)[0], np.shape(m1)[0]))
            f2 = m2
        else:
            c = np.zeros((np.shape(m1)[0], np.shape(m2)[0]))
        for k in kernel_list:
            kwidth = width_combine[k]
            f1 = m1[:, kernel_list[k]]
            if m2 is not None:
                f2 = m2[:, kernel_list[k]]
            ktype = k
            c += kernel(ktype, m1=f1, m2=f2, 
                        kwidth=kwidth,
                        kfree=kfree,
                        kdegree=kdegree)
        return c#, kwidth, ktype
    
    # Form multliplication covariance matrix.
    if combine_kernels is 'multiplication':
        if m2 is None:
            c = np.ones((np.shape(m1)[0], np.shape(m1)[0]))
            f2 = m2
        else:
            c = np.ones((np.shape(m1)[0], np.shape(m2)[0]))
        for k in kernel_list:
            kwidth = width_combine[k]
            f1 = m1[:, kernel_list[k]]
            if m2 is not None:
                f2 = m2[:, kernel_list[k]]
            ktype = k
            c *= kernel(ktype, m1=f1, m2=f2, 
                        kwidth=kwidth,
                        kfree=kfree,
                        kdegree=kdegree)
        return c#, kwidth, ktype

def dkernel_dwidth(fpm_j, width_j, ktype='gaussian'):
    """ Partial derivative of Kernel functions. """
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
    