"""Functions to prepare and return kernel data."""
import numpy as np


def prepare_kernels(kernel_dict, regularization_bounds, eval_gradients, N_D):
    """Format kernel_dictionary and stores bounds for optimization.

    Parameters
    ----------
    kernel_dict : dict
        Dictionary containing all information for the kernels.
    regularization_bounds : tuple
        Optional to change the bounds for the regularization.
    eval_gradients : boolean
        Flag to change kernel setup based on gradients being defined.
    N_D : int
        Number of dimensions of the original data.
    """
    kdict = kernel_dict
    bounds = ()

    # Set some default bounds.
    default_bounds = ((1e-12, None),)
    if eval_gradients:
        default_bounds = ((1e-6, 1e6),)

    for key in kdict:
        if 'features' in kdict[key]:
            N_D = len(kdict[key]['features'])

        if 'scaling' in kdict[key]:
            if 'scaling_bounds' in kdict[key]:
                bounds += kdict[key]['scaling_bounds']
            else:
                bounds += default_bounds

        if 'd_scaling' in kdict[key]:
            d_scaling = kdict[key]['d_scaling']
            if type(d_scaling) is float or type(d_scaling) is int:
                kernel_dict[key]['d_scaling'] = np.zeros(N_D,) + d_scaling
            if 'bounds' in kdict[key]:
                bounds += kdict[key]['bounds']
            else:
                bounds += default_bounds * N_D

        if 'width' in kdict[key] and kdict[key]['type'] != 'linear':
            theta = kdict[key]['width']
            if type(theta) is float or type(theta) is int:
                kernel_dict[key]['width'] = np.zeros(N_D,) + theta
            if 'bounds' in kdict[key]:
                bounds += kdict[key]['bounds']
            else:
                bounds += default_bounds * N_D

        elif 'hyperparameters' in kdict[key]:
            theta = kdict[key]['hyperparameters']
            if type(theta) is float or type(theta) is int:
                kernel_dict[key]['hyperparameters'] = (np.zeros(N_D,) +
                                                       theta)
            if 'bounds' in kdict[key]:
                bounds += kdict[key]['bounds']
            else:
                bounds += default_bounds * N_D

        elif 'theta' in kdict[key]:
            theta = kdict[key]['theta']
            if type(theta) is float or type(theta) is int:
                kernel_dict[key]['hyperparameters'] = (np.zeros(N_D,) +
                                                       theta)
            if 'bounds' in kdict[key]:
                bounds += kdict[key]['bounds']
            else:
                bounds += default_bounds * N_D

        elif kdict[key]['type'] == 'quadratic':
            bounds += ((1, None), (0, None),)
        elif kdict[key]['type'] == 'constant':
            theta = kdict[key]['const']
            if 'bounds' in kdict[key]:
                bounds += kdict[key]['bounds']
            else:
                bounds += default_bounds
        elif 'bounds' in kdict[key]:
            bounds += kdict[key]['bounds']

    # Bounds for the regularization
    bounds += (regularization_bounds,)

    return kernel_dict, bounds


def kdict2list(kdict, N_D=None):
    """Return ordered list of hyperparameters.

    Assumes function is given a dictionary containing properties of a single
    kernel. The dictionary must contain either the key 'hyperparameters' or
    'theta' containing a list of hyperparameters or the keys 'type' containing
    the type name in a string and 'width' in the case of a 'gaussian' or
    'laplacian' type or the keys 'degree' and 'slope' in the case of a
    'quadratic' type.

    Parameters
    ----------
    kdict : dict
        A kernel dictionary containing the keys 'type' and optional keys
        containing the hyperparameters of the kernel.
    N_D : none or int
        The number of descriptors if not specified in the kernel dict, by the
        lenght of the lists of hyperparameters.
    """

    # Get the kernel type.
    ktype = str(kdict['type'])
    if 'scaling' in kdict:
        scaling = [kdict['scaling']]
    else:
        scaling = []

    # Store hyperparameters in single list theta
    if ktype == 'gaussian' or ktype == 'sqe' or ktype == 'laplacian' and \
            'width' in kdict:
        theta = list(kdict['width'])
        if 'features' in kdict:
            N_D = len(kdict['features'])
        elif N_D is None:
            N_D = len(kdict['width'])
        if type(theta) is float:
            theta = [theta] * N_D

    # Store hyperparameters in single list theta
    if ktype == 'scaled_sqe':
        theta = list(kdict['d_scaling']) + list(kdict['width'])

    # Polynomials have pairs of hyperparamters kfree, kdegree
    elif ktype == 'quadratic':
        theta = [kdict['slope'], kdict['degree']]

    # Linear kernels have only no hyperparameters
    elif ktype == 'linear':
        theta = []

    # Constant kernel
    elif ktype == 'constant':
        theta = [kdict['const']]

    # Default hyperparameter keys for other kernels
    elif 'hyperparameters' in kdict:
        theta = kdict['hyperparameters']
        if 'features' in kdict:
            N_D = len(kdict['features'])
        elif N_D is None:
            N_D = len(theta)
        if type(theta) is float:
            theta = [theta] * N_D

    elif 'theta' in kdict:
        theta = kdict['theta']
        if 'features' in kdict:
            N_D = len(kdict['features'])
        elif N_D is None:
            N_D = len(theta)
        if type(theta) is float:
            theta = [theta]*N_D

    if 'constrained' in kdict:
        constrained = kdict['constrained']
        if 'features' in kdict:
            N_D = len(kdict['constrained'])
        elif N_D is None:
            N_D = len(constrained)
        if type(theta) is float:
            constrained = [constrained]*N_D
    else:
        constrained = []

    return scaling, theta


def kdicts2list(kernel_dict, N_D=None):
    """Return ordered list of hyperparameters given the kernel dictionary.

    The kernel dictionary must contain one or more dictionaries, each
    specifying the type and hyperparameters.

    Parameters
    ----------
    kernel_dict : dict
        A dictionary containing kernel dictionaries.
    N_D : int
        The number of descriptors if not specified in the kernel dict, by the
        length of the lists of hyperparameters.
    """
    hyperparameters = []
    for kernel_key in kernel_dict:
        theta = kdict2list(kernel_dict[kernel_key], N_D=N_D)
        hyperparameters.append(theta[0] + theta[1])
    hyperparameters = np.concatenate(hyperparameters)
    return hyperparameters


def list2kdict(hyperparameters, kernel_dict):
    """Return updated kernel dictionary with updated hyperparameters from list.

    Assumed an ordered list of hyperparametersthe and the previous kernel
    dictionary. The kernel dictionary must contain a dictionary for each kernel
    type in the same order as their respective hyperparameters in the list
    hyperparameters.

    Parameters
    ----------
    hyperparameters : list
        All hyperparameters listed in the order they are specified in the
        kernel dictionary.
    kernel_dict : dict
        A dictionary containing kernel dictionaries.
    """
    ki = 0
    for key in kernel_dict:

        ktype = kernel_dict[key]['type']

        # Retrieve the scaling factor if it is defined.
        if 'scaling' in kernel_dict[key]:
            kernel_dict[key]['scaling'] = hyperparameters[ki]
            ki += 1

        # Retreive hyperparameters from a single list theta
        if ktype == 'gaussian' or ktype == 'sqe' or ktype == 'laplacian':
            N_D = len(kernel_dict[key]['width'])
            # scaling = hyperparameters[ki]
            # kernel_dict[key]['scaling'] = scaling
            # theta = hyperparameters[ki+1:ki+1+N_D]
            theta = hyperparameters[ki:ki+N_D]
            kernel_dict[key]['width'] = list(theta)
            ki += N_D

        elif (ktype == 'scaled_sqe'):
            N_D = len(kernel_dict[key]['width'])
            kernel_dict[key]['d_scaling'] = list(hyperparameters[ki:ki+N_D])
            kernel_dict[key]['width'] = list(hyperparameters[ki+N_D:ki+2*N_D])
            ki += 2 * N_D

        # Polynomials have pairs of hyperparamters kfree, kdegree
        elif ktype == 'quadratic':
            theta = hyperparameters[ki:ki+2]
            kernel_dict[key]['slope'] = theta[0]
            kernel_dict[key]['degree'] = theta[1]
            ki += 2

        # Linear kernels have no hyperparameters
        elif ktype == 'linear':
            continue

        # If a constant is added.
        elif ktype == 'constant':
            kernel_dict[key]['const'] = hyperparameters[ki]
            ki += 1

        # Default hyperparameter keys for other kernels
        else:
            N_D = len(kernel_dict[key]['hyperparameters'])
            theta = hyperparameters[ki:ki+N_D]
            kernel_dict[key]['hyperparameters'] = list(theta)
    return kernel_dict
