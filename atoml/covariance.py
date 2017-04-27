# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:27:47 2017

"""

import numpy as np
from .kernels import kernel, kernel_combine, dkernel_dwidth

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