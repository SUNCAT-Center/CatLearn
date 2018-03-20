"""Setup a generic kernel."""
import numpy as np


def general_kernel(features, dimension='single'):
    """Generate a default kernel."""
    width = default_width(features, dimension)

    default = {
        'k1': {
            'type': 'linear', 'scaling': 1.,
        },
        'k2': {
            'type': 'constant', 'const': 1.,
        },
        'k3': {
            'type': 'gaussian', 'width': width, 'scaling': 1.,
            'dimension': dimension
        },
        'k4': {
            'type': 'quadratic', 'slope': 1., 'degree': 1., 'scaling': 1.,
            'dimension': dimension
        },
        'k5': {
            'type': 'laplacian', 'width': width, 'scaling': 1.,
            'dimension': dimension
        },
    }

    return default


def default_width(features, dimension='single'):
    """Generate defaults for the kernel width.

    Parameters
    ----------
    features : array
        The feature matrix for the training data.
    dimension : str
        The number of parameters to return. Can be 'single', or 'all'.

    Returns
    -------
    std : array
        The standard deviation of the features.
    """
    msg = 'The dimension parameter must be "single" or "all"'
    assert dimension in ['single', 'all'], msg
    axis = None
    if dimension is not 'single':
        axis = 1

    std = np.std(features, axis=axis)

    return std
