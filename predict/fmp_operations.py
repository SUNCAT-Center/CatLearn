# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 12:01:30 2017

@author: mhangaard
"""

import numpy as np


class fpm_operations(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def append_features(A, new_features):
        shapeX = np.shape(A)
        shapeX2 = np.shape(new_features)
        assert shapeX[0] == shapeX2[0]
        new_fpm = np.zeros([shapeX[0],shapeX[1]+shapeX2[1]])
        new_fpm[:,:shapeX[1]] = A
        new_fpm[:,shapeX[1]:] = new_features
        return new_fpm
    
    def get_order_2(A):
        shapeA = np.shape(A)
        nfi = 0
        new_features = np.zeros([shapeA[0],shapeA[1]**2])
        for f1 in range(shapeA[1]):
            for f2 in range(shapeA[1]):
                new_feature = A[:,f1]*A[:,f2]
                new_features[:,nfi] = new_feature
                nfi += 1
        return new_features
        
    def append_o2(self):
        new_features = self.get_order_2(self.X)
        self.append_features(self.X, new_features)