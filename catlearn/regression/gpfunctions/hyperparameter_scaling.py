"""Utility to scale hyperparameters."""
from __future__ import absolute_import

from catlearn.regression.gpfunctions.kernel_scaling import kernel_scaling


def hyperparameters(scaling, kernel_list):
    """Scale the hyperparameters."""
    return kernel_scaling(scaling, kernel_list, rescale=False)


def rescale_hyperparameters(scaling, kernel_list):
    """Rescale hyperparameters."""
    return kernel_scaling(scaling, kernel_list, rescale=True)
