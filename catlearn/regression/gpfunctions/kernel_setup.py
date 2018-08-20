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
    # Set some default bounds.
    bounds = ()
    default_bounds = ((1e-10, None),)
    if eval_gradients:
        default_bounds = ((1e-6, 1e6),)

    for key in kernel_dict:
        cN_D = N_D
        kdict = kernel_dict[key]
        msg = 'A kernel type should be set, e.g. "linear", "gaussian", etc'
        assert 'type' in kdict, msg

        if 'features' in kdict:
            cN_D, f = len(kdict['features']), cN_D
            msg = 'Trying to use greater number of features than available.'
            assert cN_D <= f, msg

        if 'dimension' in kdict:
            msg = 'Can assign parameters in "single" dimension, or in the '
            msg += 'number of "features".'
            assert kdict['dimension'] in ['single', 'features'], msg
            if kdict['dimension'] is 'single':
                cN_D = 1

        if 'scaling' in kdict:
            bounds = _scaling_setup(kdict, bounds, default_bounds)

        ktype = kdict['type']
        if ktype != 'user' and ktype != 'linear':
            cmd = '_{}_setup(kdict, bounds, cN_D, default_bounds)'.format(
                ktype)
            try:
                bounds = eval(cmd)
            except NameError:
                msg = '{} kernel not implemented'.format(ktype)
                raise NotImplementedError(msg)

    # Bounds for the regularization
    bounds += (regularization_bounds,)

    return kernel_dict, bounds


def _scaling_setup(kdict_param, bounds, default_bounds):
    msg = 'Scaling parameter should be a float.'
    assert isinstance(kdict_param['scaling'], float), msg
    if 'scaling_bounds' in kdict_param:
        bounds += kdict_param['scaling_bounds']
    else:
        bounds += default_bounds

    return bounds


def _constant_setup(kdict_param, bounds, N_D, default_bounds):
    """Setup the constant kernel."""
    allowed_keys = ['type', 'operation', 'features', 'dimension', 'const',
                    'bounds', 'scaling_bounds']
    msg1 = "An undefined key, '"
    msg2 = "', has been provided in a 'constant' type kernel dict."
    for k in kdict_param:
        assert k in allowed_keys, msg1 + k + msg2

    msg = 'Constant parameter should be a float.'
    assert isinstance(kdict_param['const'], float), msg

    if 'bounds' in kdict_param:
        bounds += kdict_param['bounds']
    else:
        bounds += default_bounds

    return bounds


def _gaussian_setup(kdict_param, bounds, N_D, default_bounds):
    """Setup the gaussian kernel."""
    msg = 'An initial width must be set.'
    assert 'width' in kdict_param, msg

    allowed_keys = ['type', 'operation', 'features', 'dimension', 'scaling',
                    'width', 'bounds', 'scaling_bounds']
    msg1 = "An undefined key, '"
    msg2 = "', has been provided in a 'gaussian' type kernel dict"
    for k in kdict_param:
        assert k in allowed_keys, msg1 + k + msg2

    # Format the width parameter.
    theta = kdict_param['width']
    try:
        theta = float(theta)
    except TypeError:
        pass
    if type(theta) is float:
        kdict_param['width'] = [theta] * N_D
    else:
        msg = 'Expected width dimensions ({}) do not match '.format(N_D)
        msg += 'those provided ({})'.format(len(theta))
        assert len(theta) == N_D, msg
    for width in kdict_param['width']:
        msg = 'Require positive widths in Gaussian kernel'
        assert width > 0., msg

    # Format the bounds if provided.
    if 'bounds' in kdict_param:
        bounds += kdict_param['bounds']
    else:
        bounds += default_bounds * N_D

    return bounds


def _quadratic_setup(kdict_param, bounds, N_D, default_bounds):
    """Setup the gaussian kernel."""
    msg = 'An initial slope and degree parameter must be set.'
    assert 'slope' in kdict_param and 'degree' in kdict_param, msg

    allowed_keys = ['type', 'operation', 'features', 'dimension', 'scaling',
                    'slope', 'degree', 'bounds', 'scaling_bounds']
    msg1 = "An undefined key, '"
    msg2 = "', has been provided in a 'quadratic' type kernel dict"
    for k in kdict_param:
        assert k in allowed_keys, msg1 + k + msg2

    # Format the slope parameter.
    theta = kdict_param['slope']
    try:
        theta = float(theta)
    except TypeError:
        pass
    if type(theta) is float:
        kdict_param['slope'] = [theta] * N_D
    else:
        msg = 'Expected slope dimensions ({}) do not match '.format(N_D)
        msg += 'those provided ({})'.format(len(theta))
        assert len(theta) == N_D, msg

    # Format the bounds if provided.
    if 'bounds' in kdict_param:
        bounds += kdict_param['bounds']
    else:
        bounds += default_bounds * (N_D + 1)

    return bounds


def _laplacian_setup(kdict_param, bounds, N_D, default_bounds):
    """Setup the laplacian kernel."""
    msg = 'An initial width must be set.'
    assert 'width' in kdict_param, msg

    allowed_keys = ['type', 'operation', 'features', 'dimension', 'scaling',
                    'width', 'bounds', 'scaling_bounds']
    msg1 = "An undefined key, '"
    msg2 = "', has been provided in a 'laplacian' type kernel dict."
    for k in kdict_param:
        assert k in allowed_keys, msg1 + k + msg2

    # Format the width parameter.
    theta = kdict_param['width']
    try:
        theta = float(theta)
    except TypeError:
        pass
    if type(theta) is float:
        kdict_param['width'] = [theta] * N_D
    else:
        msg = 'Expected width dimensions ({}) do not match '.format(N_D)
        msg += 'those provided ({})'.format(len(theta))
        assert len(theta) == N_D, msg

    # Format the bounds if provided.
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
    scaling = []
    if 'scaling' in kdict:
        scaling = [kdict['scaling']]

    if 'features' in kdict:
        N_D = len(kdict['features'])

    # Store hyperparameters in single list theta
    theta = _get_theta(kdict, ktype, N_D)

    return scaling, theta


def _get_theta(kdict, ktype, N_D):
    """Generate the theta parameter.

    Parameters
    ----------
    kdict : dict
        A kernel dictionary containing the keys 'type' and optional keys
        containing the hyperparameters of the kernel.
    ktype : str
        The kernel type.
    N_D : none or int
        The number of descriptors if not specified in the kernel dict, by the
        lenght of the lists of hyperparameters.
    """
    # Store hyperparameters in single list theta
    theta = []
    if 'width' in kdict:
        theta = list(kdict['width'])

    if 'const' in kdict:
        theta = [kdict['const']]

    if ktype == 'scaled_sqe':
        theta = list(kdict['d_scaling']) + kdict['width']

    # Polynomials have pairs of hyperparamters kfree, kdegree
    elif ktype == 'quadratic':
        theta = list(kdict['slope']) + [kdict['degree']]

    # Default hyperparameter keys for other kernels
    for param in ['hyperparameters', 'theta']:
        if param in kdict:
            theta = kdict[param]
            if N_D is None:
                N_D = len(theta)
            if type(theta) is float:
                theta = [theta] * N_D

    return theta


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
            kernel_dict[key]['scaling'] = float(hyperparameters[ki])
            ki += 1

        # Retreive hyperparameters from a single list theta
        if ktype == 'gaussian' or ktype == 'sqe' or ktype == 'laplacian':
            N_D = len(kernel_dict[key]['width'])
            # scaling = hyperparameters[ki]
            # kernel_dict[key]['scaling'] = scaling
            # theta = hyperparameters[ki+1:ki+1+N_D]
            theta = hyperparameters[ki:ki + N_D]
            kernel_dict[key]['width'] = list(theta)
            ki += N_D

        elif (ktype == 'scaled_sqe'):
            N_D = len(kernel_dict[key]['width'])
            kernel_dict[key]['d_scaling'] = list(hyperparameters[ki:ki + N_D])
            kernel_dict[key]['width'] = list(
                hyperparameters[ki + N_D:ki + 2 * N_D])
            ki += 2 * N_D

        # Quadratic have pairs of hyperparamters slope, degree
        elif ktype == 'quadratic':
            N_D = len(kernel_dict[key]['slope'])
            theta = hyperparameters[ki:ki + N_D + 1]
            kernel_dict[key]['slope'] = theta[:N_D]
            kernel_dict[key]['degree'] = theta[N_D:]
            ki += N_D + 1

        # Linear kernels have no hyperparameters
        elif ktype == 'linear':
            continue

        # If a constant is added.
        elif ktype == 'constant':
            kernel_dict[key]['const'] = float(hyperparameters[ki])
            ki += 1

        # Default hyperparameter keys for other kernels
        else:
            N_D = len(kernel_dict[key]['hyperparameters'])
            theta = hyperparameters[ki:ki + N_D]
            kernel_dict[key]['hyperparameters'] = list(theta)

    return kernel_dict
