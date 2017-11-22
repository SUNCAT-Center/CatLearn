""" Contains kernel functions and gradients of kernels. """
import numpy as np
from scipy.spatial import distance
from atoml.regression.gpfunctions.gkernels import gkernels as gkernels
from timeit import default_timer as timer


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


def constant_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    kernel_type = 'constant'
    if log_scale:
        theta = np.exp(theta)

    if eval_gradients == False:
        if m2 is None:
            m2 = m1
        return np.ones([len(m1), len(m2)]) * theta
    if eval_gradients == True:
        if m2 is None:
            k = gkernels.bigk_tilde(kernel_type,theta, m1)
            return k
        else:
            k = gkernels.k_tilde(kernel_type,theta, m1, m2)
            return k


def gaussian_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Return covariance between data m1 & m2 with a gaussian kernel.

    Parameters
    ----------
    theta : list
        A list of widths for each feature.
    log_scale : boolean
        Scaling hyperparameters in kernel can be useful for optimization.
    eval_gradients : boolean
        Analytical gradients of the training features can be included.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list
        A list of the training fingerprint vectors.
    """
    kernel_type = 'squared_exponential'
    kwidth = theta
    kwidth = np.array(kwidth)
    if log_scale:
        kwidth = np.exp(kwidth)
    if m2 is None:
        k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)


        if eval_gradients == True:
            m1 = np.array([[0.0,1.5],[1.0,1.0],[2.0,1.0]])
            print(m1)
            exit()
            start=timer()
            size = np.shape(m1)
            # big_kdd = np.zeros((size[0]*size[1],size[0]*size[1]))
            d = (m1[np.newaxis,:,:] - m1[:,np.newaxis,:])
            invsqkwidth = 1/kwidth**2
            sqd = np.sqrt(np.sum(invsqkwidth*d**2,axis=-1))
            k1 = np.exp(-(sqd)**2/2)
            big_kgd = -(invsqkwidth*d*k1[:,:,np.newaxis]).reshape(size[0],
            size[0]*size[1])
            print(big_kgd)
            # print(big_kdd1)
            # exit()
            end = timer()
            print(end-start)
            big_kdd1 = gkernels.big_kdd(kernel_type, kwidth,m1)


            k = np.block([[k,big_kgd],[np.transpose(big_kgd),big_kdd1]])
        return k
    else:
        k = distance.cdist(m1 / kwidth, m2 / kwidth, metric='sqeuclidean')
        k = np.exp(-.5 * k)
        if eval_gradients == True:
            size_m1 = np.shape(m1)
            size_m2 = np.shape(m2)
            d = (m1[np.newaxis,:,:] - m2[:,np.newaxis,:])
            invsqkwidth = 1/kwidth**2
            sqd = np.sqrt(np.sum(invsqkwidth*d**2,axis=-1))
            k1 = np.exp(-(sqd)**2/2)
            kgd_tilde = ((invsqkwidth*d*k1[:,:,np.newaxis]).swapaxes(
            size_m1[1],1).reshape(size_m2[0]*size_m2[1],size_m1[0])).T
            k = np.block([k, kgd_tilde])
        return k

def sqe_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Return covariance between data m1 & m2 with a gaussian kernel.

    Parameters
    ----------
    theta : list
        A list of widths for each feature.
    log_scale : boolean
        Scaling hyperparameters in kernel can be useful for optimization.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list
        A list of the training fingerprint vectors.
    """
    kernel_type = 'sqe'
    kwidth = theta
    if log_scale:
        kwidth = np.exp(kwidth)

    if eval_gradients == False:
        if m2 is None:
            k = distance.pdist(m1, metric='seuclidean', V=kwidth)
            k = distance.squareform(np.exp(-.5 * k))
            np.fill_diagonal(k, 1)
            return k
        else:
            k = distance.cdist(m1, m2, metric='seuclidean', V=kwidth)
            return np.exp(-.5 * k)

    if eval_gradients == True:
        msg = 'Evaluation of the gradients for this kernel is not yet ' \
        'implemented'
        raise NotImplementedError(msg)

def scaled_sqe_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Return covariance between data m1 & m2 with a gaussian kernel.

    Parameters
    ----------
    theta : list
        A list of hyperparameters.
    log_scale : boolean
        Scaling hyperparameters in kernel can be useful for optimization.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list
        A list of the training fingerprint vectors.
    """
    kernel_type = 'scaled_sqe'
    N_D = len(theta) / 2
    scaling = np.vstack(theta[:N_D])
    kwidth = np.vstack(theta[N_D:])
    if log_scale:
        scaling, kwidth = np.exp(scaling), np.exp(kwidth)

    if eval_gradients == False:
        if m2 is None:
            m2 = m1
        return distance.cdist(m1, m2,
                              lambda u, v: scaling * np.exp(np.sqrt((u -
                              v)**2 /kwidth)))
    if eval_gradients == True:
        msg = 'Evaluation of the gradients for this kernel is not yet ' \
        'implemented'
        raise NotImplementedError(msg)


def AA_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Return covariance between data m1 & m2 with Aichinson & Aitken kernel.

    Parameters
    ----------
    theta : list
        [l, n, c]
    log_scale : boolean
        Scaling hyperparameters in kernel can be useful for optimization.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list
        A list of the training fingerprint vectors.
    """
    kernel_type = 'AA'
    l = theta[0]
    c = np.vstack(theta[1:])
    if log_scale:
        l, c = np.exp(l), np.exp(c)

    if eval_gradients == False:
        n = np.shape(m1)[1]
        q = (1 - l)/(c - l)
        if m2 is None:
            m2 = m1
        return distance.cdist(m1, m2, lambda u, v:
                              (l ** (n - np.sqrt(((u - v) ** 2))) * (q **
                              np.sqrt((u - v) ** 2))).sum())

    if eval_gradients == True:
        msg = 'Evaluation of the gradients for this kernel is not yet ' \
        'implemented'
        raise NotImplementedError(msg)


def linear_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Return covariance between data m1 & m2 with a linear kernel.

    Parameters
    ----------
    theta : list
        A list containing constant offset.
    log_scale : boolean
        Scaling hyperparameters in kernel can be useful for optimization.
    eval_gradients : boolean
        Analytical gradients of the training features can be included.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list or None
        A list of the training fingerprint vectors.
    """
    kernel_type = 'linear'
    kwidth = theta
    if log_scale:
        kwidth = np.exp(kwidth)

    if eval_gradients == False:
        if m2 is None:
            m2 = m1
        return np.inner(m1, m2)

    if eval_gradients == True:
        if m2 is None:
            k = gkernels.bigk_tilde(kernel_type,kwidth, m1)
            return k
        else:
            k = gkernels.k_tilde(kernel_type,kwidth, m1, m2)
            return k



def quadratic_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Return covariance between data m1 & m2 with a quadratic kernel.

    Parameters
    ----------
    theta : list
        A list containing slope and degree for quadratic.
    log_scale : boolean
        Scaling hyperparameters in kernel can be useful for optimization.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list or None
        A list of the training fingerprint vectors.
    """
    kernel_type = 'quadratic'
    slope = theta[0]
    degree = theta[1]
    if log_scale:
        slope, degree = np.exp(slope), np.exp(degree)

    if eval_gradients == False:
        if m2 is None:
            k = distance.pdist(m1 / slope*degree, metric='sqeuclidean')
            k = distance.squareform((1. + .5*k)**-degree)
            np.fill_diagonal(k, 1)
            return k

        else:
            k = distance.cdist(m1 / slope*degree, m2 / slope*degree,
            metric='sqeuclidean')
            return (1. + .5*k)**-degree

    if eval_gradients == True:
        msg = 'Evaluation of the gradients for this kernel is not yet ' \
        'implemented'
        raise NotImplementedError(msg)


def laplacian_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Return covariance between data m1 & m2 with a laplacian kernel.

    Parameters
    ----------
    theta : list
        A list of widths for each feature.
    log_scale : boolean
        Scaling hyperparameters in kernel can be useful for optimization.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list or None
        A list of the training fingerprint vectors.
    """
    kernel_type = 'laplacian'
    if log_scale:
        theta = np.exp(theta)


    if eval_gradients == False:
        if m2 is None:
            k = distance.pdist(m1 / theta, metric='cityblock')
            k = distance.squareform(np.exp(-k))
            np.fill_diagonal(k, 1)
            return k

        else:
            k = distance.cdist(m1 / theta, m2 / theta, metric='cityblock')
            return np.exp(-k)

    if eval_gradients == True:
        msg = 'Evaluation of the gradients for this kernel is not yet ' \
        'implemented'
        raise NotImplementedError(msg)

