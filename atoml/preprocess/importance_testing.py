"""Functions to check feature significance."""
from __future__ import absolute_import
from __future__ import division

import numpy as np


def feature_invariance(features, index):
    """Make a feature invariant.

    Parameters
    ----------
    features : array
        The original feature matrix.
    index : int
        The index of the feature to be transformed.

    Returns
    -------
    features : array
        Feature matrix with an invariant feature column in matrix.
    """
    f = features.copy()
    m = np.mean(f[:, index])
    s = np.std(f[:, index])
    f[:, index] = m + s

    return f


def feature_randomize(features, index):
    """Make a feature random noise.

    Parameters
    ----------
    features : array
        The original feature matrix.
    index : int
        The index of the feature to be transformed.

    Returns
    -------
    features : array
        Feature matrix with an invariant feature column in matrix.
    """
    f = features.copy()
    m = np.mean(f[:, index])
    s = np.std(f[:, index])
    r = np.random.random(np.shape(f)[0])
    f[:, index] = (r * s) + m

    return f


def feature_shuffle(features, index):
    """Shuffle a feature.

    The method has a number of advantages for measuring feature importance.
    Notably the original values and scale of the feature are maintained.

    Parameters
    ----------
    features : array
        The original feature matrix.
    index : int
        The index of the feature to be shuffled.

    Returns
    -------
    features : array
        Feature matrix with a shuffled feature column in matrix.
    """
    f = features.copy()
    np.random.shuffle(f[:, index])

    return f
