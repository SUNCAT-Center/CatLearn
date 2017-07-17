# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard

This example script requires that the make_fingerprints.py has already been run
or that the user has generated a feature matrix in fpm.txt.
"""
import numpy as np

data = np.genfromtxt('fpm.txt')

shape = np.shape(data)

energy = data[:, -1]
train_matrix = data[:, :-1]
print('Delete me.')