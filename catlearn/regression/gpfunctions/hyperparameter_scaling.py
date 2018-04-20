"""Utility to scale hyperparameters."""
from __future__ import absolute_import

from atoml.regression.gpfunctions.kernel_scaling import kernel_scaling


def hyperparameters(scaling, kernel_dict):
    """Scale the hyperparameters."""
    return kernel_scaling(scaling, kernel_dict, rescale=False)


def rescale_hyperparameters(scaling, kernel_dict):
    """Rescale hyperparameters."""
    return kernel_scaling(scaling, kernel_dict, rescale=True)
