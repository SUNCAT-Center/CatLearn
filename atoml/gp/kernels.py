""" Contains kernel functions and gradients of kernels. """
import numpy as np
from scipy.spatial import distance


def kdict2list(kdict, N_D=None):
    """ Returns ordered list of hyperparameters, given a dictionary containing
        properties of a single kernel. The dictionary must contain either the
        key 'hyperparameters' or 'theta' containing a list of hyperparameters
        or the keys 'type' containing the type name in a string and 'width' in
        the case of a 'gaussian' or 'laplacian' type or the keys 'kdegree' and
        'kfree' in the case of a 'polynomial' type.

        Parameters
        ----------
        kdict : dict
            A kernel dictionary containing the keys 'type' and optional
            keys containing the hyperparameters of the kernel.
        N_D : none or int
            The number of descriptors if not specified in the kernel dict,
            by the lenght of the lists of hyperparameters.
    """
    # Get the kernel type.
    ktype = str(kdict['type'])
    if 'scaling' in kdict:
        scaling = [kdict['scaling']]
    else:
        scaling = []

    # Store hyperparameters in single list theta
    if (ktype == 'gaussian' or ktype == 'sqe' or ktype == 'laplacian') and \
            'width' in kdict:
        # theta = [kdict['scaling']] + list(kdict['width'])
        theta = list(kdict['width'])
        if 'features' in kdict:
            N_D = len(kdict['features'])
        elif N_D is None:
            N_D = len(kdict['width'])
        if type(theta) is float:
            theta = [theta]*N_D

    # Polynomials have pairs of hyperparamters kfree, kdegree
    elif ktype == 'polynomial':
        # kfree = kernel_dict[key]['kfree']
        # kdegree = kernel_dict[key]['kdegree']
        theta = [kdict['slope'], kdict['degree'], kdict['const']]
        # if type(kfree) is float:
        #    kfree = np.zeros(N_D,) + kfree
        # if type(kdegree) is float:
        #    kdegree = np.zeros(N_D,) + kdegree
        # zipped_theta = zip(kfree,kdegree)
        # Pass them in order [kfree1, kdegree1, kfree2, kdegree2,...]
        # theta = [hp for k in zipped_theta for hp in k]
    # Linear kernels have no hyperparameters
    elif ktype == 'linear':
        theta = [kdict['const']]

    # Default hyperparameter keys for other kernels
    elif 'hyperparameters' in kdict:
        theta = kdict['hyperparameters']
        if 'features' in kdict:
            N_D = len(kdict['features'])
        elif N_D is None:
            N_D = len(theta)
        if type(theta) is float:
            theta = [theta]*N_D

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

    return scaling, theta  # , constrained


def kdicts2list(kernel_dict, N_D=None):
    """ Returns an ordered list of hyperparameters given the kernel dictionary.
        The kernel dictionary must contain one or more dictionaries, each
        specifying the type and hyperparameters.

        Parameters
        ----------
        kernel_dict : dict
            A dictionary containing kernel dictionaries.
        N_D : int
            The number of descriptors if not specified in the kernel dict,
            by the length of the lists of hyperparameters.
    """
    hyperparameters = []
    for kernel_key in kernel_dict:
        theta = kdict2list(kernel_dict[kernel_key], N_D=N_D)
        hyperparameters.append(theta[0] + theta[1])
    hyperparameters = np.concatenate(hyperparameters)
    return hyperparameters


def list2kdict(hyperparameters, kernel_dict):
    """ Returns a updated kernel dictionary with updated hyperparameters,
        given an ordered list of hyperparametersthe and the previous kernel
        dictionary. The kernel dictionary must contain a dictionary for each
        kernel type in the same order as their respective hyperparameters
        in the list hyperparameters.

        Parameters
        ----------
        hyperparameters : list
            All hyperparameters listed in the order they are specified
            in the kernel dictionary.
        kernel_dict : dict
            A dictionary containing kernel dictionaries.
    """
    ki = 0
    for key in kernel_dict:
        ktype = kernel_dict[key]['type']
        if 'scaling' in kernel_dict[key]:
            kernel_dict[key]['scaling'] = hyperparameters[ki]
            ki += 1

        # Retreive hyperparameters from a single list theta
        if (ktype == 'gaussian' or ktype == 'sqe' or ktype == 'laplacian'):
            N_D = len(kernel_dict[key]['width'])
            # scaling = hyperparameters[ki]
            # kernel_dict[key]['scaling'] = scaling
            # theta = hyperparameters[ki+1:ki+1+N_D]
            theta = hyperparameters[ki:ki+N_D]
            kernel_dict[key]['width'] = list(theta)
            ki += N_D

        # Polynomials have pairs of hyperparamters kfree, kdegree
        elif ktype == 'polynomial':
            theta = hyperparameters[ki:ki+3]
            kernel_dict[key]['slope'] = theta[0]
            kernel_dict[key]['degree'] = theta[1]
            kernel_dict[key]['const'] = theta[2]
            ki += 3

        # Linear kernels have no hyperparameters
        elif ktype == 'linear':
            theta = hyperparameters[ki:ki+1]
            kernel_dict[key]['const'] = theta[0]
            ki += 1

        # Default hyperparameter keys for other kernels
        else:
            N_D = len(kernel_dict[key]['hyperparameters'])
            theta = hyperparameters[ki:ki+N_D]
            kernel_dict[key]['hyperparameters'] = list(theta)
    return kernel_dict


def gaussian_kernel(theta, m1, m2=None):
    """ Returns the covariance matrix between datasets m1 and m2
        with a gaussian kernel.

        Parameters
        ----------
        theta : list
            A list of widths for each feature.
        m1 : list
            A list of the training fingerprint vectors.
        m2 : list
            A list of the training fingerprint vectors.
    """
    # scaling = theta[0]
    kwidth = theta  # [1:]
    if m2 is None:
        k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)
        return k
        # return scaling * k
    else:
        k = distance.cdist(m1 / kwidth, m2 / kwidth,
                           metric='sqeuclidean')
        return np.exp(-.5 * k)
        # return scaling * np.exp(-.5 * k)


def sqe_kernel(theta, m1, m2=None):
    """ Returns the covariance matrix between datasets m1 and m2
        with a gaussian kernel.

        Parameters
        ----------
        theta : list
            A list of widths for each feature.
        m1 : list
            A list of the training fingerprint vectors.
        m2 : list
            A list of the training fingerprint vectors.
    """
    # scaling = theta[0]
    kwidth = theta  # [1:]
    if m2 is None:
        k = distance.pdist(m1, metric='seuclidean', V=kwidth)
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)
        # return scaling * k
        return k
    else:
        k = distance.cdist(m1, m2,
                           metric='seuclidean', V=kwidth)
        return np.exp(-.5 * k)
        # return scaling * np.exp(-.5 * k)


def AA_kernel(theta, m1, m2=None):
    """ Returns the covariance matrix between datasets m1 and m2
        with an Aichinson & Aitken kernel.

        Parameters
        ----------
        theta : list
            [l, n, c]
        m1 : list
            A list of the training fingerprint vectors.
        m2 : list
            A list of the training fingerprint vectors.
    """
    if m2 is None:
        m2 = m1
    l = theta[0]
    c = np.vstack(theta[1:])
    n = np.shape(m1)[1]
    q = (1 - l)/(c - l)
    return distance.cdist(m1, m2, lambda u, v:
                          (l ** (n - np.sqrt(((u - v) ** 2))) *
                           (q ** np.sqrt((u - v) ** 2))).sum())


def linear_kernel(theta, m1, m2=None):
    """ Returns the covariance matrix between datasets m1 and m2
        with a linear kernel.

        Parameters
        ----------
        theta : list
            A list containing constant offset.
        m1 : list
            A list of the training fingerprint vectors.
        m2 : list or None
            A list of the training fingerprint vectors.
    """
    if m2 is None:
        m2 = m1
    c = np.zeros([len(m1), len(m2)]) + theta
    return np.inner(m1, m2) + c


def polynomial_kernel(theta, m1, m2=None):
    """ Returns the covariance matrix between datasets m1 and m2
        with a polynomial kernel.

        Parameters
        ----------
        theta : list
            A list containing constant, slope and degree for polynomial.
        m1 : list
            A list of the training fingerprint vectors.
        m2 : list or None
            A list of the training fingerprint vectors.
    """
    slope = theta[0]
    degree = theta[1]
    const = theta[2]
    if m2 is None:
        m2 = m1
    return (const + np.dot(m1, np.transpose(m2)) * slope) ** degree


def laplacian_kernel(theta, m1, m2=None):
    """ Returns the covariance matrix between datasets m1 and m2
        with a laplacian kernel.

        Parameters
        ----------
        theta : list
            A list of widths for each feature.
        m1 : list
            A list of the training fingerprint vectors.
        m2 : list or None
            A list of the training fingerprint vectors.
    """
    if m2 is None:
        k = distance.pdist(m1 / theta, metric='cityblock')
        k = distance.squareform(np.exp(-k))
        np.fill_diagonal(k, 1)
        return k
    else:
        k = distance.cdist(m1 / theta, m2 / theta, metric='cityblock')
        return np.exp(-k)
