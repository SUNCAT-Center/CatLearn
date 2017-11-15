# created on 11/9/17 at 12:59
# by @jagarridotorres (jagt@stanford.edu)

import numpy as np
from scipy.spatial import distance

# Squared exponential gaussian kernel:
def kernel_k(l, xm, xn):
    d = xm - xn
    k = np.exp(-np.linalg.norm(d/l)**2/2)
    return k


def kernel_kgd(l, xm, xn):
    d = xm - xn
    kgd = l**(-2) * d * np.exp(-np.linalg.norm(d/l)**2/2)
    return kgd


def kernel_kdd(l, xm, xn):
    assert len(xm) == len(xn)
    I_m = np.identity(len(xm))
    d = xm - xn
    kdd = (I_m*l**(-2) - np.outer(l**(-2)*d,(l**(-2)*d).T))*np.exp(-np.linalg.norm(
    d/l)**2/2)
    return kdd


def kernel_klittle(l, test, train):
    k = distance.cdist(train / l, test / l, metric='sqeuclidean')
    return np.exp(-.5 * k)


def kernel_bigk(l, train):
    k = distance.pdist(train / l, metric='sqeuclidean')
    k = distance.squareform(np.exp(-.5 * k))
    np.fill_diagonal(k, 1)
    return k
