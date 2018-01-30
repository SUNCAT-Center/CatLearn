"""Function performing uncertainty analysis."""
from __future__ import absolute_import
from __future__ import division

import numpy as np

from .covariance import get_covariance


def get_uncertainty(kernel_dict, test_fp, reg, ktb, cinv, log_scale,
                    include_noise=True):
    """Function to calculate uncertainty.

    Parameters
    ----------
    kernel_dict : dict
        Dictionary containing all information for the kernels.
    test_fp : array
        Test feature set.
    reg : float
        Regularization parameter.
    ktb : array
        Covariance matrix for test and training data.
    cinv : array
        Covariance matrix for training dataset.
    log_scale : boolean
        Flag to define if the hyperparameters are log scale.
    include_noise : boolean
        Flag to determine whether to calculate the uncertainty on the full GP,
        including the noise on the data. Default is True.

    Returns
    -------
    uncertainty : list
        The uncertainty on each prediction in the test data. By default, this
        includes a measure of the noise on the data.
    """
    # Set noise to zero if this shouldn't be accounted for.
    if not include_noise:
        reg = 0.

    # Generate the test covariance matrix.
    kxx = get_covariance(
        kernel_dict=kernel_dict, matrix1=test_fp, log_scale=log_scale,
        eval_gradients=False
    )

    # Calculate the prediction variance for test data.
    scale = np.diagonal(kxx)
    var = np.einsum("ij,ij->i", np.dot(ktb, cinv), ktb)
    uncertainty = np.sqrt(reg + scale - var)

    return uncertainty
