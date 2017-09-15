"""Functions to calculate the cost statistics."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from collections import defaultdict


def get_error(prediction, target, epsilon=None, return_percentiles=True):
    """Return error for predicted data.

    Discussed in: Rosasco et al, Neural Computation, (2004), 16, 1063-1076.

    Parameters
    ----------
    prediction : list
        A list of predicted values.
    target : list
        A list of target values.
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

    # Epsilon-insensitive error function.
    if epsilon is not None:
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
