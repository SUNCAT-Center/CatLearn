# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 12:01:30 2017

@author: mhangaard
"""

import numpy as np

def triangular(n):
    return sum(range(n+1))

class fpm_operations():
    def __init__(self, X=None, x=None):
        self.X = X
        self.x = x
   
    def get_order_2(self):
        A = self.X
        """Get all combinations x_ij = x_i * x_j, where x_i,j are features.
        The sorting order in dimension 0 is preserved.
        Input)
            A: nxm matrix, where n is the number of training examples and 
            m is the number of features.
        Output)
            n x m**2 matrix
        """
        shapeA = np.shape(A)
        nfi = 0
        new_features = np.zeros([shapeA[0],triangular(shapeA[1])])
        for f1 in range(shapeA[1]):
            for f2 in range(f1,shapeA[1]):
                new_feature = A[:,f1]*A[:,f2]
                new_features[:,nfi] = new_feature
                nfi += 1
        return new_features
    
    def get_labels_order_2(self):
        """Get all combinations ij, where i,j are feature labels.
        Input)
            x: length m vector, where m is the number of features.
        Output)
            m**2 vector
        """
        L = len(self.x)
        #assert L == np.shape(self.X)[1]
        new_features = []
        for f1 in range(L):
            for f2 in range(f1,L):
                new_features.append(self.x[f1]+'_x_'+self.x[f2])
        return np.array(new_features)
    
    def get_order_2ab(self, a, b):
        A = self.X
        """Get all combinations x_ij = x_i*a * x_j*b, where x_i,j are features.
        The sorting order in dimension 0 is preserved.
        Input)
            A: nxm matrix, where n is the number of training examples and 
            m is the number of features.
            
            a: float
            
            b: float
        Output)
            n x m**2 matrix
        """
        shapeA = np.shape(A)
        nfi = 0
        new_features = np.zeros([shapeA[0],triangular(shapeA[1])])
        for f1 in range(shapeA[1]):
            for f2 in range(f1,shapeA[1]):
                new_feature = A[:,f1]**a * A[:,f2]**b
                new_features[:,nfi] = new_feature
                nfi += 1
        return new_features
    
    def get_ablog(self,a,b):
        A = self.X
        """Get all combinations x_ij = a*log(x_i) + b*log(x_j), 
        where x_i,j are features.
        The sorting order in dimension 0 is preserved.
        Input)
            A: nxm matrix, where n is the number of training examples and 
            m is the number of features.
            
            a: float
            
            b: float
        Output)
            n x m**2 matrix
        """
        shapeA = np.shape(A)
        nfi = 0
        new_features = np.zeros([shapeA[0],triangular(shapeA[1])])
        for f1 in range(shapeA[1]):
            for f2 in range(f1,shapeA[1]):
                new_feature = a*np.log(A[:,f1]) + b*np.log(A[:,f2])
                new_features[:,nfi] = new_feature
                nfi += 1
        return new_features        
    
    def fpmatrix_split(self, nsplit):
        """ Routine to split list of candidates into sublists. This can be
            useful for bootstrapping, LOOCV, etc.
            Input:
                nsplit: int
                The number of bins that data should be divided into.
            Output:
                list
        """
        dataset = []
        np.random.shuffle(self.X)
        # Calculate the number of items per split.
        n = len(self.X) / nsplit
        # Get any remainders.
        r = len(self.X) % nsplit
        # Define the start and finish of first split.
        s1 = 0
        s2 = n + min(1, r)
        for _ in range(nsplit):
            dataset.append(self.X[int(s1):int(s2),:])
            # Get any new remainder.
            r = max(0, r-1)
            # Define next split.
            s1 = s2
            s2 = s2 + n + min(1, r)
        return dataset