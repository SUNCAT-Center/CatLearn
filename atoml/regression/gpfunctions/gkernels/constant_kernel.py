# created on 11/9/17 at 12:59
# by @jagarridotorres (jagt@stanford.edu)

import numpy as np

# Constant kernel:

def kernel_k(l, xm, xn):
    k = 1.0
    return k


def kernel_kgd(l, xm, xn):
    kgd = 1.0
    return kgd


def kernel_kdd(l, xm, xn):
    kdd = np.ones((len(xm),len(xm)))
    return kdd