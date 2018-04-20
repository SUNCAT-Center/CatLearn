"""Function to compute Sammon's error between original and reduced features."""
from scipy.spatial import distance
import numpy as np


def sammons_error(original, reduced):
    """Sammon error.

    Parameters
    ----------
    original : array
        The original feature set.
    reduced : array
        The reduced feature set.

    Returns
    -------
    error : float
        Sammon's error value.
    """
    # Calculate Euclidean distances.
    original_dist = distance.squareform(
        distance.pdist(original, metric='euclidean'))
    reduced_dist = distance.squareform(
        distance.pdist(reduced, metric='euclidean'))

    # Setup i < j part.
    original_tri = np.tril(original_dist, -1)
    reduced_tri = np.tril(reduced_dist, -1)

    # Expect division by zero warnings.
    with np.errstate(invalid='ignore'):
        dist_err = np.true_divide(
            (original_tri - reduced_tri) ** 2, original_tri)
    # Replace NAN values from zero division
    dist_err[np.isnan(dist_err)] = 0.

    # Calculate the actual error.
    error = (1. / original_tri.sum()) * dist_err.sum()

    return error
