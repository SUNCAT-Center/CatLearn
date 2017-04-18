# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 12:01:30 2017

@author: mhangaard
"""
import numpy as np
from scipy import cluster
from random import shuffle
from collections import defaultdict

from .feature_select import sure_independence_screening


def triangular(n):
    return sum(range(n+1))


def do_sis(X, y, size=None, increment=1):
    """ function to narrow down a list of descriptors based on sure
        independence screening.

        Parameters
        ----------
        X : array
            n x m matrix
        y : list
            Length n vector
        l : list
            Length m list of strings (optional)
        size : integer
            (optional)
        increment : integer
            (optional)

        Returns
        -------
        l : list
            List of s surviving indices.

        Example
        -------
            l = do_sis(X,y)
            X[:,l]
            will produce the fingerprint matrix using only surviving
            descriptors.
    """
    shape = np.shape(X)
    l = np.arange(shape[1])
    if size is None:
        size = shape[0]
    while shape[1] >= size:
        shape = np.shape(X)
        select = sure_independence_screening(y, X, size=shape[1]-increment)
        X = X[:, select['accepted']]
        l = l[select['accepted']]
    return l


def get_order_2(A):
    """ Get all combinations x_ij = x_i * x_j, where x_i,j are features. The
        sorting order in dimension 0 is preserved.

        Parameters
        ----------
        A : array
            n x m matrix, where n is the number of training examples and m is
            the number of features.

        Returns
        -------
        n x triangular(m) matrix
    """
    shapeA = np.shape(A)
    nfi = 0
    new_features = np.zeros([shapeA[0], triangular(shapeA[1])])
    for f1 in range(shapeA[1]):
        for f2 in range(f1, shapeA[1]):
            new_feature = A[:, f1]*A[:, f2]
            new_features[:, nfi] = new_feature
            nfi += 1
    return new_features


def get_div_order_2(A):
    """ Get all combinations x_ij = x_i / x_j, where x_i,j are features. The
        sorting order in dimension 0 is preserved. If a value is 0, Inf is
        returned.

        Parameters
        ----------
        A : array
            n x m matrix, where n is the number of training examples and m is
            the number of features.

        Returns
        -------
        n x m**2 matrix
    """
    shapeA = np.shape(A)
    nfi = 0
    # Preallocate:
    new_features = np.zeros([shapeA[0], shapeA[1]**2])
    for f1 in range(shapeA[1]):
        for f2 in range(shapeA[1]):
            new_feature = np.true_divide(A[:, f1], A[:, f2])
            new_features[:, nfi] = new_feature
            nfi += 1
    return new_features


def get_labels_order_2(l, div=False):
    """ Get all combinations ij, where i,j are feature labels.

        Parameters
        ----------
        x : list
            Length m vector, where m is the number of features.

        Returns
        -------
        m**2 vector or triangular(m) vector
    """
    L = len(l)
    new_features = []
    if div:
        op = '_div_'
        s = 0
    else:
        op = '_x_'
    for f1 in range(L):
        if not div:
            s = f1
        for f2 in range(s, L):
            new_features.append(l[f1] + op + l[f2])
    return new_features


def get_order_2ab(A, a, b):
    """ Get all combinations x_ij = x_i*a * x_j*b, where x_i,j are features. The
        sorting order in dimension 0 is preserved.

        Parameters
        ----------
        A : array
            n x m matrix, where n is the number of training examples and m is
            the number of features.
        a : float

        b : float

        Returns
        -------
        n x triangular(m) matrix
    """
    shapeA = np.shape(A)
    nfi = 0
    new_features = np.zeros([shapeA[0], triangular(shapeA[1])])
    for f1 in range(shapeA[1]):
        for f2 in range(f1, shapeA[1]):
            new_feature = A[:, f1]**a * A[:, f2]**b
            new_features[:, nfi] = new_feature
            nfi += 1
    return new_features


def get_labels_order_2ab(l, a, b):
    """ Get all combinations ij, where i,j are feature labels.

        Parameters
        ----------
        x : list
            Length m vector, where m is the number of features.

        Returns
        -------
        m**2 vector or triangular(m) vector
    """
    L = len(l)
    new_features = []
    for f1 in range(L):
        for f2 in range(f1, L):
            new_features.append(l[f1] + '_' + str(a) + '_x_' + l[f2] + '_' +
                                str(b))
    return new_features


def get_ablog(A, a, b):
    """ Get all combinations x_ij = a*log(x_i) + b*log(x_j), where x_i,j are
        features. The sorting order in dimension 0 is preserved.

        Parameters
        ----------
        A : array
            An n x m matrix, where n is the number of training examples and
            m is the number of features.
        a : float

        b : float

        Returns
        -------
        n x triangular(m) matrix
    """
    shapeA = np.shape(A)
    shift = np.abs(np.min(A, axis=0)) + 1.
    A += shift
    nfi = 0
    new_features = np.zeros([shapeA[0], triangular(shapeA[1])])
    for f1 in range(shapeA[1]):
        for f2 in range(f1, shapeA[1]):
            new_feature = a*np.log(A[:, f1]) + b*np.log(A[:, f2])
            new_features[:, nfi] = new_feature
            nfi += 1
    return new_features


def get_labels_ablog(l, a, b):
    """ Get all combinations ij, where i,j are feature labels.

        Parameters
        ----------
        a : float
        b : float

        Returns
        -------
        m ** 2 vector or triangular(m) vector
    """
    L = len(l)
    new_features = []
    for f1 in range(L):
        for f2 in range(f1, L):
            # TODO Better string formatting with numbers.
            new_features.append('log' + str(a) + '_' + l[f1] + 'log' + str(b) +
                                '_' + l[f2])
    return new_features


def fpmatrix_split(X, nsplit, fix_size=None, replacement=False):
    """ Routine to split feature matrix and return sublists. This can be
        useful for bootstrapping, LOOCV, etc.

        Parameters
        ----------
        nsplit : int
            The number of bins that data should be devided into.
        fix_size : int
            Define a fixed sample size, e.g. nsplit=5 fix_size=100, generates
            5 x 100 data split. Default is None, all avaliable data is divided
            nsplit times.
        replacement : boolean
            Set true to generate samples with replacement e.g. a candidate can
            be in multiple samles. Default is False.
    """
    if fix_size is not None:
        msg = 'Cannot divide dataset in this way, number of candidates is '
        msg += 'too small'
        assert len(X) >= nsplit * fix_size, msg
    dataset = []
    index = list(range(len(X)))
    shuffle(index)
    # Find the size of the divides based on all candidates.
    s1 = 0
    if fix_size is None:
        # Calculate the number of items per split.
        n = len(X) / nsplit
        # Get any remainders.
        r = len(X) % nsplit
        # Define the start and finish of first split.
        s2 = n + min(1, r)
    else:
        s2 = fix_size
    for _ in range(nsplit):
        if replacement:
            shuffle(index)
        dataset.append(X[index[int(s1):int(s2)]])
        s1 = s2
        if fix_size is None:
            # Get any new remainder.
            r = max(0, r-1)
            # Define next split.
            s2 = s2 + n + min(1, r)
        else:
            s2 = s2 + fix_size
    return dataset


def cluster_features(train_matrix, train_target, k=2, test_matrix=None,
                     test_target=None):
    """ Function to perform k-means clustering in the feature space. """
    m = defaultdict(list)

    centroids, order = cluster.vq.kmeans2(train_matrix, k)
    # Generate a list of colors for the training data.
    c = []
    for i in range(k):
        c.append([float(i)/float(k), float(i)/float(k)/float(k-i),
                  float(k-i)/float(k)])  # R,G,B
    m['colors'] = ([c[i] for i in order])

    # Break up the training data based on clusters.
    split_f = {}
    split_t = {}
    for f, t, l in zip(train_matrix, train_target, order):
        if l not in split_f:
            split_f[l] = []
            split_t[l] = []
        split_f[l].append(f)
        split_t[l].append(t)
    m['train_features'] = split_f
    m['train_target'] = split_t

    # Cluster test data based on training centroids.
    for t, tt in zip(test_matrix, test_target):
        td = float('inf')
        for i in range(len(centroids)):
            d = np.linalg.norm(t - centroids[i])
            if d < td:
                mini = i
                td = d
        m['test_order'].append(mini)

    # Break up the test data based on clusters.
    if test_matrix is not None:
        if test_target is not None:
            test_f = {}
            test_t = {}
            for f, t, l in zip(test_matrix, test_target, m['test_order']):
                if l not in test_f:
                    test_f[l] = []
                    test_t[l] = []
                test_f[l].append(f)
                test_t[l].append(t)
            m['test_features'] = test_f
            m['test_target'] = test_t
        else:
            test_f = {}
            for f, t, l in zip(test_matrix, m['test_order']):
                if l not in test_f:
                    test_f[l] = []
                test_f[l].append(f)
            m['train_features'] = test_f

    return m
