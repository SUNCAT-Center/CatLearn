"""Functions to calculate the cost statistics."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from collections import defaultdict


def get_error(prediction, target, epsilon=None):
    """Return error for predicted data.

    Discussed in: Rosasco et al, Neural Computation, (2004), 16, 1063-1076.

    Parameters
    ----------
    prediction : list
        A list of predicted values.
    target : list
        A list of target values.
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
    error['signed_average'] = np.average(res)

    # Root mean squared error function.
    e_sq = np.square(res)
    error['rmse_all'] = np.sqrt(e_sq)
    error['rmse_average'] = np.sqrt(np.sum(e_sq)/len(e_sq))

    # Absolute error function.
    e_abs = np.abs(res)
    error['absolute_all'] = e_abs
    error['absolute_average'] = np.sum(e_abs)/len(e_abs)

    # Epsilon-insensitive error function.
    if epsilon is not None:
        e_epsilon = np.abs(res) - epsilon
        np.place(e_epsilon, e_epsilon < 0, 0)
        error['insensitive_all'] = e_epsilon
        error['insensitive_average'] = np.sum(e_epsilon)/len(e_epsilon)

    return error
