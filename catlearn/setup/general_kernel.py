"""Setup a generic kernel."""
import numpy as np


def general_kernel(features, dimension='single'):
    """Generate a default kernel."""
    length = default_lengthscale(features, dimension)

    default = {
        'k1': {
            'type': 'linear', 'scaling': 1.,
        },
        'k2': {
            'type': 'constant', 'const': 1.,
        },
        'k3': {
            'type': 'gaussian', 'width': length, 'scaling': 1.,
            'dimension': dimension
        },
        'k4': {
            'type': 'quadratic', 'slope': length, 'degree': 1., 'scaling': 1.,
            'dimension': dimension
        },
        'k5': {
            'type': 'laplacian', 'width': length, 'scaling': 1.,
            'dimension': dimension
        },
    }

    return default


def default_lengthscale(features, dimension='single'):
    """Generate defaults for the kernel lengthscale.

    Parameters
    ----------
    features : array
        The feature matrix for the training data.
    dimension : str
        The number of parameters to return. Can be 'single', or 'features'.

    Returns
    -------
    std : array
        The standard deviation of the features.
    """
    msg = 'The dimension parameter must be "single" or "features"'
    assert dimension in ['single', 'features'], msg
    axis = None
    if dimension is not 'single':
        axis = 0

    std = np.std(features, axis=axis)

    return std
