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
                            scale_optimizer, eval_gradients, eval_jac=False):
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
    L = cholesky(K, overwrite_a=False, lower=True, check_finite=True)
    a = cho_solve((L, True), y, check_finite=False)
    datafit = -.5 * np.dot(y.T, a)
    complexity = -np.log(np.diag(L)).sum()  # (A.18) in R. & W.
    normalization = -.5 * n * np.log(2 * np.pi)

    # Get the log marginal likelihood.
    p = (datafit + complexity + normalization).sum()
    if not eval_jac:
        return -p
    else:
        # Get jacobian of log marginal likelyhood wrt. hyperparameters.
        C = cho_solve((L, True), np.eye(n), check_finite=True)
        aa = a*a.T  # inner1d(a,a)
        Q = aa - C
        # Get the list of gradients.
        jac = dK_dtheta_j(theta, train_matrix, kernel_dict, Q)
        # Append gradient with respect to regularization.
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
    K = get_covariance(
        kernel_dict=kernel_dict, matrix1=train_matrix,
        regularization=theta[-1], log_scale=scale_optimizer,
        eval_gradients=eval_gradients)
    n = len(targets)
    y = targets.reshape([n, 1])
    L = cholesky(K, overwrite_a=False, lower=True, check_finite=True)
    a = cho_solve((L, True), y, overwrite_b=False, check_finite=True)
    C = cho_solve((L, True), np.eye(n), check_finite=True)
    aa = a*a.T  # inner1d(a,a)
    Q = aa - C
    # Get the list of gradients.
    jac = dK_dtheta_j(theta, train_matrix, kernel_dict, Q)
    # Append gradient with respect to regularization.
    dKdnoise = np.identity(n)
    jac.append(0.5*np.sum(inner1d(Q, dKdnoise.T)))
    return -np.array(jac)


def dK_dtheta_j(theta, train_matrix, kernel_dict, Q):
    jac = []
    N, N_D = np.shape(train_matrix)
    ki = 0
    for key in kernel_dict.keys():
        kdict = kernel_dict[key]
        ktype = kdict['type']
        if 'scaling' in kdict or kdict['type'] == 'gaussian':
            scaling, hyperparameters = kdict2list(kdict, N_D)
            k = eval(
                'ak.{}_kernel(m1=train_matrix, m2=None, theta=hyperparameters, \
                eval_gradients=False, log_scale=False)'.format(ktype))
        if 'scaling' in kdict:
            jac.append(0.5*np.sum(inner1d(Q, k.T)))
            ki += 1
        if kdict['type'] == 'constant':
            jac.append(0.5*np.sum(inner1d(Q, np.ones([N, N]))))
            ki += 1
        elif ktype == 'linear':
            continue
        elif kdict['type'] == 'gaussian':
            kwidth = theta[ki:ki+N_D]
            m = len(kwidth)
            for j in range(0, m):
                dKdtheta_j = ak.gaussian_dk_dwidth(k, j, train_matrix,
                                                   kwidth)
                if 'scaling' in kdict:
                    dKdtheta_j *= scaling
                jac.append(0.5*np.sum(inner1d(Q, dKdtheta_j.T)))
            ki += N_D
        elif kdict['type'] == 'quadratic':
            slope = theta[ki:ki+N_D]
            m = len(slope)
            for j in range(0, m):
                dKdtheta_j = ak.quadratic_dk_dslope(k, j, train_matrix,
                                                    slope)
                if 'scaling' in kdict:
                    dKdtheta_j *= scaling
                jac.append(0.5*np.sum(inner1d(Q, dKdtheta_j.T)))
            degree = theta[ki+N_D]
            dKdtheta_degree = ak.quadratic_dk_ddegree(k, j, train_matrix,
                                                      degree)
            jac.append(0.5*np.sum(inner1d(Q, dKdtheta_degree.T)))
            ki += N_D + 1
        else:
            raise NotImplementedError("jacobian for " + ktype)
    return jac
