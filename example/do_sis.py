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
from atoml.fpm_operations import get_order_2, do_sis

fpm_raw = np.genfromtxt('fpm.txt')

shape = np.shape(fpm_raw)


energy = fpm_raw[:, -2]

fpm_ml0 = fpm_raw[:, :-2]

# Operations to generate new descriptors
order2 = get_order_2(fpm_ml0)
fpm_ml = np.hstack([fpm_ml0,order2])

shape = np.shape(fpm_ml)
print(shape)

print('Performing single step SIS reducing a',shape[1],'descriptors')
select = sure_independence_screening(energy, fpm_ml, size=shape[0])
print('to', len(select['accepted']), 'descriptors:')
print(select['accepted'])

increment = 10
print('Performing interative SIS reducing ',shape[1],' descriptors, in steps of',increment)
l = do_sis(fpm_ml, energy, shape[0], increment)
print('to', len(l), 'descriptors:')
print(l)