"""Functions to calculate the cost statistics."""
from __future__ import absolute_import
from __future__ import division
import functools
import numpy as np
from collections import defaultdict
from .gpfunctions.covariance import get_covariance
from .gpfunctions.kernel_setup import list2kdict


def get_error(prediction, target, metrics=None, epsilon=None,
              return_percentiles=True):
    """Return error for predicted data.

    Discussed in: Rosasco et al, Neural Computation, (2004), 16, 1063-1076.

    Parameters
    ----------
    prediction : list
        A list of predicted values.
    target : list
        A list of target values.
    metrics : list
        Define a list of additional cost functions to be returned. Can
        currently be 'log' and 'insensitive'.
    epsilon : float
        insensitivity value.
    return_percentiles : boolean
        Return some percentile statistics with the predictions.
    """
    msg = 'Something has gone wrong and there are '
    if len(prediction) < len(target):
        msg += 'more targets than predictions.'
    elif len(prediction) > len(target):
        msg += 'fewer targets than predictions.'
    assert len(prediction) == len(target), msg

    if metrics is None:
        metrics = []

    error = defaultdict(list)
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    # Residuals
    res = prediction - target
    error['residuals'] = res
    error['signed_mean'] = np.mean(res)
    error['signed_median'] = np.median(res)
    error['signed_percentiles'] = _get_percentiles(res)

    # Root mean squared error function.
    e_sq = np.square(res)
    error['rmse_all'] = np.sqrt(e_sq)
    error['rmse_average'] = np.sqrt(np.mean(e_sq))
    error['rmse_percentiles'] = _get_percentiles(error['rmse_all'])

    # Absolute error function.
    e_abs = np.abs(res)
    error['absolute_all'] = e_abs
    error['absolute_average'] = np.mean(e_abs)
    error['absolute_percentiles'] = _get_percentiles(error['absolute_all'])

    if 'log' in metrics:
        # Root mean squared logarithmic error.
        error['log_all'] = (
            np.log(np.max([prediction, np.zeros(len(prediction))], axis=0) + 1)
            - np.log(target + 1)) ** 2.
        error['rmsle_all'] = np.sqrt(error['log_all'])
        error['rmsle_average'] = np.sqrt(np.nanmean(error['log_all']))

    if 'insensitive' in metrics:
        # Epsilon-insensitive error function.
        msg = 'epsilon insensitivity must be defined, currently: {}'.format(
            epsilon)
        assert isinstance(epsilon, float)
        e_epsilon = np.abs(res) - epsilon
        np.place(e_epsilon, e_epsilon < 0, 0)
        error['insensitive_all'] = e_epsilon
        error['insensitive_average'] = np.mean(e_epsilon)
        if return_percentiles:
            error['insensitive_percentiles'] = _get_percentiles(
                error['insensitive_all'])

    return error


def _get_percentiles(residuals):
    """Get percentiles for the calculated errors.

    Parameters
    ----------
    residuals : list
        List of calculated errors.
    """
    data = {}

    # Some hard coded percentiles to return.
    percentiles = [99, 95, 75, 25, 5, 1]

    for p in percentiles:
        data['{0}'.format(p)] = np.percentile(residuals, p)

    return data


def _cost_function(theta, train_matrix, targets, kernel_dict,
                   scale_optimizer, lf):
    """Return cost function on the training data.

    Parameters
    ----------

    """
    # Make a new covariance matrix with the given hyperparameters.
    kernel_dict = list2kdict(theta, kernel_dict)
    cvm = get_covariance(kernel_dict=kernel_dict,
                         matrix1=train_matrix,
                         regularization=theta[-1],
                         log_scale=scale_optimizer,
                         eval_gradients=False)
    # Invert the covariance matrix.
    cinv = np.linalg.inv(cvm)
    # Calculate predictions for the training data.
    # Form list of the actual predictions.
    alpha = functools.reduce(np.dot, (cinv, targets))
    prediction = functools.reduce(np.dot, (cvm, alpha))
    # Calculated the error for the prediction on the training data.
    train_err = get_error(prediction=prediction,
                          target=targets)[lf + '_average']
    return train_err
