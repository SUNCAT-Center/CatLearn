"""Log marginal likelihood calculator function."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.linalg import cholesky, cho_solve
from .covariance import get_covariance
from .kernel_setup import list2kdict, kdict2list
from numpy.core.umath_tests import inner1d
from atoml.regression.gpfunctions import kernels as ak


def log_marginal_likelihood(theta, train_matrix, targets, kernel_dict,
                            scale_optimizer, eval_gradients, eval_jac):
    """Return the negative of the log marginal likelyhood.

    Equation 5.8 in C. E. Rasmussen and C. K. I. Williams, 2006

    Parameters
    ----------
    theta : list
        A list containing the hyperparameters.
    train_matrix : list
        A list of the test fingerprint vectors.
    targets : list
        A list of target values
    kernel_dict: dict
        A dictionary of kernel dictionaries
    scale_optimizer : boolean
        Flag to define if the hyperparameters are log scale for optimization.
    """
    # Get the covariance matrix.
    kernel_dict = list2kdict(theta, kernel_dict)
    K = get_covariance(
        kernel_dict=kernel_dict, matrix1=train_matrix,
        regularization=theta[-1], log_scale=scale_optimizer,
        eval_gradients=eval_gradients)
    # Setup the data.
    n = len(targets)
    y = targets.reshape([n, 1])

    # Calculate the various terms in likelihood.
    L = cholesky(K, lower=True)
    a = cho_solve((L, True), y)
    datafit = -.5*np.dot(y.T, a)
    complexity = -np.log(np.diag(L)).sum()  # (A.18) in R. & W.
    normalization = -.5*n*np.log(2*np.pi)

    # Get the log marginal likelihood.
    p = (datafit + complexity + normalization).sum()
    if not eval_jac:
        return -p
    else:
        C = cho_solve((L, True), np.eye(n))
        aa = a*a.T  # inner1d(a,a)
        Q = aa - C
        jac = dK_dtheta_j(theta, train_matrix, kernel_dict, Q)
        dKdnoise = np.identity(n)
        jac.append(0.5*np.sum(inner1d(Q, dKdnoise.T)))
        return -p, -np.array(jac)


def dlml(theta, train_matrix, targets, kernel_dict,
         scale_optimizer, eval_gradients, eval_jac=True):
    """ Return the gradient of the log marginal likelyhood.
        (Equation 5.9 in C. E. Rasmussen and C. K. I. Williams, 2006)

        Input:
            train_fp: n x m matrix
            K: n x n positive definite matrix
            widths: vector of length m
            noise: float
            y: vector of length n
    """
    kwidth = theta[:-1]
    K = get_covariance(
        kernel_dict=kernel_dict, matrix1=train_matrix,
        regularization=theta[-1], log_scale=scale_optimizer,
        eval_gradients=eval_gradients)
    m = len(kwidth)
    n = len(targets)
    y = targets.reshape([n, 1])
    L = cholesky(K, lower=True)
    a = cho_solve((L, True), y)
    C = cho_solve((L, True), np.eye(n))
    aa = a*a.T  # inner1d(a,a)
    Q = aa - C
    jac = []
    for j in range(0, m):
        dKdtheta_j = ak.gaussian_dk_dtheta(train_matrix[:, j], kwidth[j])
        jac.append(0.5*np.sum(inner1d(Q, dKdtheta_j.T)))
    dKdnoise = np.identity(n)
    jac.append(0.5*np.sum(inner1d(Q, dKdnoise.T)))
    return -np.array(jac)


def dK_dtheta_j(theta, train_matrix, kernel_dict, Q):
    jac = []
    N_D = np.shape(train_matrix)[1]
    for key in kernel_dict.keys():
        kdict = kernel_dict[key]
        ktype = kdict['type']
        if 'scaling' in kdict:
            scaling, hyperparameters = kdict2list(kdict, N_D)
            k = eval(
                'ak.{}_kernel(m1=train_matrix, m2=None, theta=hyperparameters, \
                eval_gradients=False, log_scale=False)'.format(ktype))
            jac.append(k)
        if kdict['type'] == 'gaussian':
            if 'scaling' in kdict:
                kwidth = theta[1:-1]
            else:
                kwidth = theta[:-1]
            m = len(kwidth)
            for j in range(0, m):
                dKdtheta_j = ak.gaussian_dk_dtheta(train_matrix[:, j],
                                                   kwidth[j])
                jac.append(0.5*np.sum(inner1d(Q, dKdtheta_j.T)))
        if kdict['type'] == 'constant':
                jac.append(1)
    return jac
