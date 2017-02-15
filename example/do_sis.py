# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard

This example script requires that the make_fingerprints.py has already been run
or that the user has generated a feature matrix in fpm.txt.
"""
from __future__ import print_function

import numpy as np
from atoml.fingerprint_setup import sure_independence_screening
# from predict.fpm_operations import fpm_operations

fpm_raw = np.genfromtxt('fpm.txt')

shape = np.shape(fpm_raw)
print(shape)

energy = fpm_raw[:, -2]

fpm_ml0 = fpm_raw[:, :-2]

# Operations to generate new descriptors
# ops = fpm_operations(fpm_ml0)
# order2 = ops.get_order_2()
# fpm_ml = np.hstack([fpm_ml0,order2])

select = sure_independence_screening(energy, fpm_ml0)
print(select['sorted'])
print(select['correlation'])
