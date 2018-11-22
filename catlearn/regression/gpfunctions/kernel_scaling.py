"""Function to scale kernel hyperparameters."""
from __future__ import absolute_import
from __future__ import division


def kernel_scaling(scale_data, kernel_list, rescale):
    """Base hyperparameter scaling function.

    Parameters
    ----------
    scale_data : object
        Output from the default scaling function.
    kernel_list : list
        Dictionary containing all dictionaries for the kernels.
    rescale : boolean
       Flag for whether to scale or rescale the data.
    """
    # Get relevant scaling data.
    store_mean = scale_data.feature_data['mean']
    store_std = scale_data.feature_data['std']

    for k in kernel_list:
        mean, std = store_mean, store_std
        # Check hyperparameter dimensions.
        if 'dimension' in k and k['dimension'] == 'single':
            print('single hyperparameter being used, cant scale')
            continue

        # Check feature dimensions.
        if 'features' in k:
            mean = mean[k['features']]
            std = std[k['features']]

        ktype = k['type']
        kernel = eval(
            '_{}_kernel_scale(k, mean, std, rescale)'.format(
                ktype))
        k = kernel

    return kernel_list


def _constant_kernel_scale(kernel, mean, std, rescale):
    """Scale Gaussian kernel hyperparameters.

    Parameters
    ----------
    kernel : dict
        Dictionary containing all information for a single kernel.
    mean : array
       An array of mean values for all features.
    std : array
       An array of standard deviation values for all features.
    rescale : boolean
       Flag for whether to scale or rescale the data.
    """
    print('{} kernel hyperparameters left unscaled'.format(kernel['type']))
    return kernel


def _gaussian_kernel_scale(kernel, mean, std, rescale):
    """Scale Gaussian kernel hyperparameters.

    Parameters
    ----------
    kernel : dict
        Dictionary containing all information for a single kernel.
    mean : array
       An array of mean values for all features.
    std : array
       An array of standard deviation values for all features.
    rescale : boolean
       Flag for whether to scale or rescale the data.
    """
    if not rescale:
        kernel['width'] /= std
    else:
        kernel['width'] *= std
    return kernel


def _AA_kernel_scale(kernel, mean, std, rescale):
    """Scale Gaussian kernel hyperparameters.

    Parameters
    ----------
    kernel : dict
        Dictionary containing all information for a single kernel.
    mean : array
       An array of mean values for all features.
    std : array
       An array of standard deviation values for all features.
    rescale : boolean
       Flag for whether to scale or rescale the data.
    """
    print('{} kernel hyperparameters left unscaled'.format(kernel['type']))
    return kernel


def _linear_kernel_scale(kernel, mean, std, rescale):
    """Scale Gaussian kernel hyperparameters.

    Parameters
    ----------
    kernel : dict
        Dictionary containing all information for a single kernel.
    mean : array
       An array of mean values for all features.
    std : array
       An array of standard deviation values for all features.
    rescale : boolean
       Flag for whether to scale or rescale the data.
    """
    print('{} kernel hyperparameters left unscaled'.format(kernel['type']))
    return kernel


def _quadratic_kernel_scale(kernel, mean, std, rescale):
    """Scale Gaussian kernel hyperparameters.

    Parameters
    ----------
    kernel : dict
        Dictionary containing all information for a single kernel.
    mean : array
       An array of mean values for all features.
    std : array
       An array of standard deviation values for all features.
    rescale : boolean
       Flag for whether to scale or rescale the data.
    """
    print('{} kernel hyperparameters left unscaled'.format(kernel['type']))
    return kernel


def _laplacian_kernel_scale(kernel, mean, std, rescale):
    """Scale Gaussian kernel hyperparameters.

    Parameters
    ----------
    kernel : dict
        Dictionary containing all information for a single kernel.
    mean : array
       An array of mean values for all features.
    std : array
       An array of standard deviation values for all features.
    rescale : boolean
       Flag for whether to scale or rescale the data.
    """
    print('{} kernel hyperparameters left unscaled'.format(kernel['type']))
    return kernel
