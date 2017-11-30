# created on 11/9/17 at 12:59
# by @jagarridotorres (jagt@stanford.edu)

import numpy as np

# Constant kernel:

def kernel_k(l, xm, xn):
    k = 1.0
    return k


def kernel_kgd(l, xm, xn):
    kgd = 0.0
    return kgd


def kernel_kdd(l, xm, xn):
    kdd = np.zeros((len(xm),len(xm)))
    return kdd


def kernel_klittle(l, test, train):
    return np.ones([len(train), len(test)]) * l


def kernel_bigk(l, train):
    return np.ones([len(train), len(train)]) * l
