"""Function to scale target values."""
import numpy as np
from collections import defaultdict


def target_standardize(target):
    """Return a list of standardized target values.

    Parameters
    ----------
    target : list
        A list of the target values.
    """
    target = np.asarray(target)

    data = defaultdict(list)
    data['mean'] = np.mean(target, axis=0)
    data['std'] = np.std(target, axis=0)
    data['target'] = (target - data['mean']) / data['std']
    return data


def target_normalize(target):
    """Return a list of normalized target values.

    Parameters
    ----------
    target : list
        A list of the target values.
    """
    target = np.asarray(target)

    data = defaultdict(list)
    data['mean'] = np.mean(target, axis=0)
    data['dif'] = np.max(target, axis=0) - np.min(target, axis=0)
    data['target'] = (target - data['mean']) / data['dif']

    return data


def target_center(target):
    """Return a list of normalized target values.

    Parameters
    ----------
    target : list
        A list of the target values.
    """
    target = np.asarray(target)

    data = defaultdict(list)
    data['mean'] = np.mean(target, axis=0)
    data['target'] = target - data['mean']

    return data
