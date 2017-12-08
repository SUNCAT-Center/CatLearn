"""Log marginal likelihood calculator function."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.linalg import cholesky, cho_solve

from .covariance import get_covariance
from .kernels import list2kdict


def log_marginal_likelihood(theta, train_matrix, targets, kernel_dict,
                            scale_optimizer, eval_gradients):
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
        eval_gradients=eval_gradients
        )

    # Setup the data.
    n = len(targets)
    y = targets.reshape([n, 1])

    # Calculate the various terms in likelyhood.
    L = cholesky(K, lower=True)
    a = cho_solve((L, True), y)
    datafit = -.5*np.dot(y.T, a)
    complexity = -np.log(np.diag(L)).sum()  # (A.18) in R. & W.
    normalization = -.5*n*np.log(2*np.pi)

    # Get the log marginal likelihood.
    p = (datafit + complexity + normalization).sum()

    return -p
