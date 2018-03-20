"""Contains kernel functions and gradients of kernels."""
import numpy as np
from scipy.spatial import distance


def constant_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Return constant to add to the kernel.

    Parameters
    ----------
    theta : list
        A list of widths for each feature.
    log_scale : boolean
        Scaling hyperparameters in the kernel can be useful for optimization.
    eval_gradients : boolean
        Analytical gradients of the training features can be included.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list
        A list of the training fingerprint vectors.

    Returns
    -------
    k : array
        The covariance matrix.
    """
    if log_scale:
        theta = np.exp(theta)

    # Check if gradients are evaluated.
    if not eval_gradients:
        if m2 is None:
            m2 = m1
        return np.ones([len(m1), len(m2)]) * theta

    # Account for gradients in constant kernel.
    size_m1 = np.shape(m1)
    if m2 is None:
        k = np.zeros((size_m1[0] + size_m1[0] * size_m1[1],
                      size_m1[0] + size_m1[0] * size_m1[1]))
        k[0:size_m1[0], 0:size_m1[0]] = np.ones([len(m1), len(m1)]) * theta
    else:
        size_m2 = np.shape(m2)
        k = np.zeros((size_m1[0], size_m2[0] + size_m2[0] * size_m2[1]))
        k[0:size_m1[0], 0:size_m2[0]] = np.ones([size_m1[0], size_m2[0]]) \
            * theta

    return k


def gaussian_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Generate the covariance between data with a Gaussian kernel.

    Parameters
    ----------
    theta : list
        A list of widths for each feature.
    log_scale : boolean
        Scaling hyperparameters in the kernel can be useful for optimization.
    eval_gradients : boolean
        Analytical gradients of the training features can be included.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list
        A list of the training fingerprint vectors.

    Returns
    -------
    k : array
        The covariance matrix.
    """
    kwidth = np.array(theta)

    if log_scale:
        kwidth = np.exp(kwidth)

    if m2 is None:
        k = distance.pdist(m1 / kwidth, metric='sqeuclidean')
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)

        if eval_gradients:
            return gaussian_xx_gradients(m1, kwidth, k)

    else:
        k = distance.cdist(m1 / kwidth, m2 / kwidth, metric='sqeuclidean')
        k = np.exp(-.5 * k)

        if eval_gradients:
            return gaussian_xxp_gradients(m1, m2, kwidth, k)

    return k


def gaussian_xx_gradients(m1, kwidth, k):
    """Gradient for k(x, x).

    Parameters
    ----------
    m1 : array
        Feature matrix.
    kwidth : list
        List of lengthscales for the gaussian kernel.
    k : array
        Upper left portion of the overall covariance matrix.
    """
    size = np.shape(m1)
    big_kgd = np.zeros((size[0], size[0] * size[1]))
    big_kdd = np.zeros((size[0] * size[1], size[0] * size[1]))
    invsqkwidth = kwidth**(-2)
    I_m = np.identity(size[1]) * invsqkwidth
    for i in range(size[0]):
        ldist = (invsqkwidth * (m1[:, :] - m1[i, :]))
        big_kgd_i = ((ldist).T * k[i]).T
        big_kgd[:, (size[1] * i):(size[1] + size[1] * i)] = big_kgd_i
        if size[1] <= 30:  # Broadcasting requires large memory.
            k_dd = ((I_m - (ldist[:, None, :] * ldist[:, :, None])) *
                    (k[i, None, None].T)).reshape(-1, size[1])
            big_kdd[:, size[1] * i:size[1] + size[1] * i] = k_dd
        elif size[1] > 30:  # Loop when large number of features.
            for j in range(i, size[0]):
                k_dd = (I_m - np.outer(ldist[j], ldist[j].T)) * k[i, j]
                big_kdd[i * size[1]:(i + 1) * size[1],
                        j * size[1]:(j + 1) * size[1]] = k_dd
                if j != i:
                    big_kdd[j * size[1]:(j + 1) * size[1],
                            i * size[1]:(i + 1) * size[1]] = k_dd.T

    return np.block([[k, big_kgd], [np.transpose(big_kgd), big_kdd]])


def gaussian_xxp_gradients(m1, m2, kwidth, k):
    """Gradient for k(x, x').

    Parameters
    ----------
    m1 : array
        Feature matrix.
    m2 : array
        Feature matrix typically associated with the test data.
    kwidth : list
        List of lengthscales for the gaussian kernel.
    k : array
        Upper left portion of the overall covariance matrix.
    """
    size_m1 = np.shape(m1)
    size_m2 = np.shape(m2)
    kgd_tilde = np.zeros((size_m1[0], size_m2[0] * size_m2[1]))
    invsqkwidth = kwidth**(-2)
    for i in range(size_m1[0]):
        kgd_tilde_i = -((invsqkwidth * (m2[:, :] - m1[i, :]) *
                         k[i, :].reshape(size_m2[0], 1)).reshape(
                             1, size_m2[0] * size_m2[1])
                        )
        kgd_tilde[i, :] = kgd_tilde_i

    return np.block([k, kgd_tilde])


def gaussian_dk_dwidth(k, m1, kwidth, log_scale=False):
    """Return gradient of the gaussian kernel with respect to the j'th width.

    Parameters
    ----------
    k : array
        n by n array. The (not scaled) gaussian kernel.
    m1 : list
        A list of the training fingerprint vectors.
    kwidth : float
        The full list of widths
    log_scale : boolean
        Scaling hyperparameters in kernel can be useful for optimization.
    """
    if log_scale:
        raise NotImplementedError("Log scale hyperparameters in jacobian.")
    if len(kwidth) == 1:
        dkdw = distance.pdist(m1, metric='sqeuclidean')
        dkdw = distance.squareform(dkdw / (kwidth[0] ** 3))
        np.fill_diagonal(dkdw, 0)
        dkdw *= k
        return dkdw[..., np.newaxis]
    dkdw = (m1[:, np.newaxis, :] - m1[np.newaxis, :, :]) ** 2 / (kwidth ** 3)
    # Chain rule.
    dkdw *= k[..., np.newaxis]

    return dkdw


def sqe_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Generate the covariance between data with a Gaussian kernel.

    Parameters
    ----------
    theta : list
        A list of widths for each feature.
    log_scale : boolean
        Scaling hyperparameters in the kernel can be useful for optimization.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list
        A list of the training fingerprint vectors.

    Returns
    -------
    k : array
        The covariance matrix.
    """
    if eval_gradients:
        msg = 'Evaluation of the gradients for this kernel is not yet '
        msg += 'implemented'
        raise NotImplementedError(msg)

    kwidth = theta
    if log_scale:
        kwidth = np.exp(kwidth)

    if m2 is None:
        k = distance.pdist(m1, metric='seuclidean', V=kwidth)
        k = distance.squareform(np.exp(-.5 * k))
        np.fill_diagonal(k, 1)
    else:
        k = distance.cdist(m1, m2, metric='seuclidean', V=kwidth)
        k = np.exp(-.5 * k)

    return k


def scaled_sqe_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Generate the covariance between data with a Gaussian kernel.

    Parameters
    ----------
    theta : list
        A list of hyperparameters.
    log_scale : boolean
        Scaling hyperparameters in the kernel can be useful for optimization.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list
        A list of the training fingerprint vectors.

    Returns
    -------
    k : array
        The covariance matrix.
    """
    if eval_gradients:
        msg = 'Evaluation of the gradients for this kernel is not yet '
        msg += 'implemented'
        raise NotImplementedError(msg)

    N_D = len(theta) / 2
    scale = np.vstack(theta[:N_D])
    kwidth = np.vstack(theta[N_D:])
    if log_scale:
        scale, kwidth = np.exp(scale), np.exp(kwidth)

    if m2 is None:
        m2 = m1
    k = distance.cdist(
        m1, m2, lambda u, v: scale * np.exp(np.sqrt((u - v)**2 / kwidth)))

    return k


def AA_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Generate the covariance between data with a Aichinson & Aitken kernel.

    Parameters
    ----------
    theta : list
        [l, n, c]
    log_scale : boolean
        Scaling hyperparameters in the kernel can be useful for optimization.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list
        A list of the training fingerprint vectors.

    Returns
    -------
    k : array
        The covariance matrix.
    """
    if eval_gradients:
        msg = 'Evaluation of the gradients for this kernel is not yet '
        msg += 'implemented'
        raise NotImplementedError(msg)

    ll = theta[0]
    c = np.vstack(theta[1:])
    if log_scale:
        ll, c = np.exp(ll), np.exp(c)

    n = np.shape(m1)[1]
    q = (1 - ll) / (c - ll)
    if m2 is None:
        m2 = m1
    k = distance.cdist(
        m1, m2, lambda u, v: (ll ** (n - np.sqrt(((u - v) ** 2))) *
                              (q ** np.sqrt((u - v) ** 2))).sum())

    return k


def linear_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Generate the covariance between data with a linear kernel.

    Parameters
    ----------
    theta : list
        A list containing constant offset.
    log_scale : boolean
        Scaling hyperparameters in the kernel can be useful for optimization.
    eval_gradients : boolean
        Analytical gradients of the training features can be included.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list or None
        A list of the training fingerprint vectors.

    Returns
    -------
    k : array
        The covariance matrix.
    """
    if not eval_gradients:
        if m2 is None:
            m2 = m1
        return np.inner(m1, m2)

    if m2 is None:
        return np.block(
            [[np.inner(m1, m1), np.tile(m1, len(m1))],
             [np.transpose(np.tile(m1, len(m1))),
              np.ones([np.shape(m1)[0] * np.shape(m1)[1],
                       np.shape(m1)[0] * np.shape(m1)[1]])]])
    else:
        return np.block([[np.inner(m1, m2), np.tile(m1, len(m2))]])


def quadratic_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Generate the covariance between data with a quadratic kernel.

    Parameters
    ----------
    theta : list
        A list containing slope and degree for quadratic.
    log_scale : boolean
        Scaling hyperparameters in the kernel can be useful for optimization.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list or None
        A list of the training fingerprint vectors.

    Returns
    -------
    k : array
        The covariance matrix.
    """
    if eval_gradients:
        msg = 'Evaluation of the gradients for this kernel is not yet '
        msg += 'implemented'
        raise NotImplementedError(msg)

    slope = np.array(theta[0])
    degree = theta[1]
    if log_scale:
        slope, degree = np.exp(slope), np.exp(degree)

    if m2 is None:
        k = distance.pdist(m1 / slope * degree, metric='sqeuclidean')
        k = distance.squareform((1. + .5 * k)**-degree)
        np.fill_diagonal(k, 1)

    else:
        k = distance.cdist(m1 / slope * degree, m2 / slope * degree,
                           metric='sqeuclidean')
        k = (1. + .5 * k)**-degree

    return k


def quadratic_dk_dslope(k, m1, slope, log_scale=False):
    raise NotImplementedError("Quadratic kernel jacobian wrt. slope.")


def quadratic_dk_ddegree(k, m1, degree, log_scale=False):
    raise NotImplementedError("Quadratic kernel jacobian wrt. degree.")


def laplacian_kernel(theta, log_scale, m1, m2=None, eval_gradients=False):
    """Generate the covariance between data with a laplacian kernel.

    Parameters
    ----------
    theta : list
        A list of widths for each feature.
    log_scale : boolean
        Scaling hyperparameters in the kernel can be useful for optimization.
    m1 : list
        A list of the training fingerprint vectors.
    m2 : list or None
        A list of the training fingerprint vectors.

    Returns
    -------
    k : array
        The covariance matrix.
    """
    if eval_gradients:
        msg = 'Evaluation of the gradients for this kernel is not yet '
        msg += 'implemented'
        raise NotImplementedError(msg)

    theta = np.array(theta)
    if log_scale:
        theta = np.exp(theta)

    if m2 is None:
        k = distance.pdist(m1 / theta, metric='cityblock')
        k = distance.squareform(np.exp(-k))
        np.fill_diagonal(k, 1)
    else:
        k = distance.cdist(m1 / theta, m2 / theta, metric='cityblock')
        k = np.exp(-k)

    return k


def laplacian_dk_dwidth(k, m1, kwidth, log_scale=False):
    # raise NotImplementedError("Laplacian kernel jacobian wrt. width.")
    if log_scale:
        raise NotImplementedError("Log scale hyperparameters in jacobian.")
    if len(kwidth) == 1:
        dkdw = distance.pdist(m1 / (kwidth[0] ** 2), metric='cityblock')
        dkdw = distance.squareform(dkdw)
        np.fill_diagonal(dkdw, 0)
        dkdw *= k
        return dkdw[..., np.newaxis]
    dkdw = abs(m1[:, np.newaxis, :] - m1[np.newaxis, :, :]) / (kwidth ** 2)
    # Chain rule.
    dkdw *= k[..., np.newaxis]
    return dkdw
