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
        msg = 'A kernel type should be set, e.g. "linear", "gaussian", etc'
        assert 'type' in kdict[key], msg

        if 'features' in kdict[key]:
            N_D = len(kdict[key]['features'])

        if 'scaling' in kdict[key]:
            bounds = _scaling_setup(kdict_param=kdict[key], bounds=bounds,
                                    default_bounds=default_bounds)

        ktype = kdict[key]['type']
        if ktype is not 'user' and ktype is not 'linear':
            cmd = '_{0}_setup(kdict[key], bounds, N_D, default_bounds)'.format(
                kdict[key]['type'])
            try:
                bounds = eval(cmd)
            except NameError:
                msg = '{} kernel not implemented'.format(kdict[key]['type'])
                raise NotImplementedError(msg)

    # Bounds for the regularization
    bounds += (regularization_bounds,)

    return kernel_dict, bounds


def _scaling_setup(kdict_param, bounds, default_bounds):
    msg = 'Scaling parameter should be a float.'
    assert type(kdict_param['scaling']) is float, msg
    if 'scaling_bounds' in kdict_param:
        bounds += kdict_param['scaling_bounds']
    else:
        bounds += default_bounds

    return bounds


def _linear_setup(kdict_param, bounds, N_D, default_bounds):
    """Setup the linear kernel."""


def _constant_setup(kdict_param, bounds, N_D, default_bounds):
    """Setup the linear kernel."""
    if 'bounds' in kdict_param:
        bounds += kdict_param['bounds']
    else:
        bounds += default_bounds

    return bounds


def _gaussian_setup(kdict_param, bounds, N_D, default_bounds):
    """Setup the gaussian kernel."""
    theta = kdict_param['width']
    if type(theta) is float or type(theta) is int:
        kdict_param['width'] = np.zeros(N_D,) + theta
    if 'bounds' in kdict_param:
        bounds += kdict_param['bounds']
    else:
        bounds += default_bounds * N_D

    return bounds


def _quadratic_setup(kdict_param, bounds, N_D, default_bounds):
    """Setup the gaussian kernel."""
    # theta = kdict_param['slope']
    # if type(theta) is float or type(theta) is int:
    #    kdict_param['slope'] = np.zeros(N_D,) + theta
    if 'bounds' in kdict_param:
        bounds += kdict_param['bounds']
    else:
        bounds += default_bounds  # * N_D

    # theta = kdict_param['degree']
    # if type(theta) is float or type(theta) is int:
    #    kdict_param['degree'] = np.zeros(N_D,) + theta
    if 'bounds' in kdict_param:
        bounds += kdict_param['bounds']
    else:
        bounds += default_bounds  # * N_D

    return bounds


def _laplacian_setup(kdict_param, bounds, N_D, default_bounds):
    """Setup the laplacian kernel."""
    theta = kdict_param['width']
    if type(theta) is float or type(theta) is int:
        kdict_param['width'] = np.zeros(N_D,) + theta
    if 'bounds' in kdict_param:
        bounds += kdict_param['bounds']
    else:
        bounds += default_bounds * N_D

    return bounds


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
