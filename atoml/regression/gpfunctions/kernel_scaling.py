"""Function to scale kernel hyperparameters."""
from __future__ import absolute_import
from __future__ import division


def kernel_scaling(scale_data, kernel_dict, rescale):
    """Base hyperparameter scaling function.

    Parameters
    ----------
    scale_data : object
        Output from the default scaling function.
    kernel_dict : dict
        Dictionary containing all information for the kernels.
    rescale : boolean
       Flag for whether to scale or rescale the data.
    """
    # Get relevant scaling data.
    store_mean = scale_data.feature_data['mean']
    store_std = scale_data.feature_data['std']

    for k in kernel_dict.keys():
        mean, std = store_mean, store_std
        # Check hyperparameter dimensions.
        if 'dimension' in kernel_dict[k] and \
           kernel_dict[k]['dimension'] == 'single':
            print('single hyperparameter being used, cant scale')
            continue

        # Check feature dimensions.
        if 'features' in kernel_dict[k]:
            mean = mean[kernel_dict[k]['features']]
            std = std[kernel_dict[k]['features']]

        ktype = kernel_dict[k]['type']
        kernel = eval(
            '_{}_kernel_scale(kernel_dict[k], mean, std, rescale)'.format(
                ktype))
        kernel_dict[k] = kernel

    return kernel_dict


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
