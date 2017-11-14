# created on 11/9/17 at 12:59
# by @jagarridotorres (jagt@stanford.edu)

import numpy as np

# Linear kernel:

def kernel_k(l, xm, xn):
    k = np.inner(xm,xn)
    return k


def kernel_kgd(l, xm, xn):
    kgd = xm
    return kgd


def kernel_kdd(l, xm, xn):
    kdd = np.ones((len(xm),len(xm)))
    return kdd