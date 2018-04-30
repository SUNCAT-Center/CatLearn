"""Log marginal likelihood calculator function."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.linalg import cholesky, cho_solve
from .covariance import get_covariance
from .kernel_setup import list2kdict, kdict2list
from catlearn.regression.gpfunctions import kernels as ak


def log_marginal_likelihood(theta, train_matrix, targets, kernel_dict,
                            scale_optimizer, eval_gradients, cinv=None,
                            eval_jac=False):
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
    eval_gradients : boolean
        Flag to specify whether to compute gradients in covariance.
    cinv : array
        Pre-computed inverted covariance matrix.
    eval_jac : boolean
        Flag to specify whether to calculate gradients for hyperparameter
        optimization.
    """
    # Setup the data.
    n = len(targets)
    y = targets.reshape([n, 1])

    # Get the covariance matrix.
    kernel_dict = list2kdict(theta, kernel_dict)
    K = get_covariance(
        kernel_dict=kernel_dict, matrix1=train_matrix,
        regularization=theta[-1], log_scale=scale_optimizer,
        eval_gradients=eval_gradients)
    L = cholesky(K, overwrite_a=False, lower=True, check_finite=True)

    # Invert the covariance matrix.
    if cinv is None:
        a = cho_solve((L, True), y, check_finite=False)
    else:
        a = np.dot(cinv, y)

    # Calculate the various terms in likelihood.
    datafit = -.5 * np.dot(y.T, a)
    complexity = -np.log(np.diag(L)).sum()  # (A.18) in R. & W.
    normalization = -n * np.log(2 * np.pi) / 2.

    # Get the log marginal likelihood.
    p = (datafit + complexity + normalization).sum(-1)
    if not eval_jac:
        return -p
    else:
        # Get jacobian of log marginal likelyhood wrt. hyperparameters.
        C = cho_solve((L, True), np.eye(n),
                      check_finite=True)
        aa = a * a.T  # np.einsum("ik,jk->ijk", a, a)
        Q = (aa - C)[:, :, np.newaxis]
        # Get the list of gradients.
        dK_dtheta = dK_dtheta_j(theta, train_matrix, kernel_dict, Q)
        jac = 0.5 * np.einsum("ijl,ijk->kl", Q, dK_dtheta)
        return -p, -np.array(jac.sum(-1))


def dK_dtheta_j(theta, train_matrix, kernel_dict, Q):
    """Return the jacobian of the log marginal likelyhood.

    This is calculated with respect to the hyperparameters, as in:
    Equation 5.9 in C. E. Rasmussen and C. K. I. Williams, 2006

    Parameters
    ----------
    theta : list
        A list containing the hyperparameters.
    train_matrix : list
        A list of the test fingerprint vectors.
    kernel_dict: dict
        A dictionary of kernel dictionaries
    Q : array.
    """
    jac = []
    N, N_D = np.shape(train_matrix)
    ki = 0
    for key in kernel_dict.keys():
        kdict = kernel_dict[key]
        ktype = kdict['type']
        if 'operation' in kernel_dict[key] and \
           kernel_dict[key]['operation'] == 'multiplication':
            msg = "jacobian of product kernels wrt. hyperparameters."
            raise NotImplementedError(msg)
        if 'scaling' in kdict or kdict['type'] == 'gaussian':
            scaling, hyperparameters = kdict2list(kdict, N_D)
            k = eval(
                'ak.{}_kernel(m1=train_matrix, m2=None, theta=hyperparameters, \
                eval_gradients=False, log_scale=False)'.format(ktype))
        if 'scaling' in kdict:
            jac.append(k[..., np.newaxis])
            ki += 1
        if kdict['type'] == 'constant':
            jac.append(np.ones([N, N, 1]))
            ki += 1
        elif ktype == 'linear':
            continue
        elif kdict['type'] == 'gaussian':
            N_W = len(kdict['width'])
            kwidth = theta[ki:ki + N_W]
            dKdtheta = ak.gaussian_dk_dwidth(k, train_matrix, kwidth)
            if 'scaling' in kdict:
                dKdtheta *= scaling
            jac.append(dKdtheta)
            ki += N_W
        elif kdict['type'] == 'quadratic':
            N_S = len(kdict['slope'])
            slope = theta[ki:ki + N_S]
            dKdslope = ak.quadratic_dk_dslope(k, train_matrix, slope)
            if 'scaling' in kdict:
                dKdslope *= scaling
            jac.append(dKdslope)
            degree = theta[ki + N_D]
            dKdtheta_degree = ak.quadratic_dk_ddegree(k, train_matrix, degree)
            jac.append(dKdtheta_degree)
            ki += N_D + 1
        elif kdict['type'] == 'laplacian':
            N_W = len(kdict['width'])
            kwidth = theta[ki:ki + N_W]
            dKdtheta = ak.laplacian_dk_dwidth(k, train_matrix, kwidth)
            if 'scaling' in kdict:
                dKdtheta *= scaling
            jac.append(dKdtheta)
            ki += N_W
        else:
            raise NotImplementedError("jacobian for " + ktype)
    # Append gradient with respect to regularization.
    dKdnoise = np.eye(N)[:, :, np.newaxis]
    jac.append(dKdnoise)
    jac = np.concatenate(jac, axis=2)
    return jac
