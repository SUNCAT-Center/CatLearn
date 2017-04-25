# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 12:01:30 2017
Modified Mon April 24 2017

@author: mhangaard
contributor: doylead
"""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy import cluster
from random import shuffle
from itertools import combinations_with_replacement
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

def _separate_list(p):
    '''
    Routine to split any list into all possible combinations of two
    lists which, combined, contain all elements.

    Inputs)
        p: list
            The list to be split

    Outputs)
        combinations: list
            A list containing num_combinations elements, each of which
            is a tuple.  Each tuple contains two elements, each of which
            is a list.  These two tuple elements have no intersection, and
            their union is p.
    '''
    num_elements = len(p)
    num_combinations = (2**num_elements - 2)/2
    key = '0%db'%num_elements
    combinations = []
    for i in range(1,num_combinations+1):
        bin_str = format(i,key)
        left = []
        right = []
        for j in range(num_elements):
            if bin_str[j]=='0':
                left.append(p[j])
            elif bin_str[j]=='1':
                right.append(p[j])
        combinations.append((left,right))
    return combinations

def _decode_key(p,key):
    '''
    Routine to decode a "key" as implemented in generate_features.
    These "keys" are used to avoid duplicate terms in the numerator and denominator

    Inputs)
        p: list
            The list of input features provided by the user
        key: string
            A string containing a composite term, where each original feature in p
            is represented by its index.

            Example:
            The term given by p[0]*p[1]*p[1]*p[4] would have the key "0*1*1*4"

    Outputs)
        p_prime: string
            A string containing the composite term as a function of the original 
            input features
    '''
    p = [str(i) for i in  p]
    elements = key.split('*')
    translated_elements = [p[int(i)] for i in elements]
    unique_elements = list(set(translated_elements))
    unique_elements.sort()
    count_elements = {}
    for ele in unique_elements:
        count_elements[ele] = 0
    for ele in translated_elements:
        count_elements[ele] += 1
    ele_list = []
    for ele in unique_elements:
        count = count_elements[ele]
        if count == 1:
            ele_list.append(ele)
        if count >= 2:
            ele_list.append(ele+'^%d'%count)
    p_prime = '*'.join(ele_list)
    return p_prime

def generate_positive_features(p,N,exclude=False,s=False):
    '''
    Routine to generate a list of polynomial combinations of variables
    in list p up to order N.
    
    Example:
    p = (a,b,c) ; N = 3

    returns (order not preserved)
    [a*a*a, a*a*b, a*a*c, a*b*b, a*b*c, a*c*c, b*b*b, b*b*c, b*c*c,
    c*c*c, a*a, a*b, a*c, b*b, b*c, c*c, a, b, c]

    Inputs)
        p: list
            Features to be combined
        N: non-negative integer
            The maximum polynomial coefficient for combinations
        exclude: bool
            Set exclude=True to avoid returning 1 to represent the
            zeroth power
        s: bool
            Set s=True to return a list of strings
            Set s=False to evaluate each element in the list

    Outputs)
        all_powers: list
            A list of combinations of the input features to meet the
            required specifications
    '''
    if N==0 and s:
        return ['1']
    elif N==0 and not s:
        return [1]
    elif N==1 and not exclude and s:
        return p+["1"]
    elif N==1 and not exclude and not s:
        return p+[1]
    if N==1 and exclude:
        return p
    else:
        all_powers = []
        p = [str(i) for i in p]
        for i in range(N,0,-1):
            thisPower = combinations_with_replacement(p,i)
            tuples = list(thisPower)
            ntuples = len(tuples)
            thisPowerFeatures = ['*'.join(tuples[j]) for j in range(ntuples)]
            if not s:
                thisPowerFeatures = [eval(j) for j in thisPowerFeatures]
            all_powers.append(thisPowerFeatures)
        if not exclude:
            if s:
                all_powers.append(['1'])
            else:
                all_powers.append([1])
        all_powers = [item for sublist in all_powers for item in sublist]
        return all_powers

def generate_features(p, max_num=2, max_den=1, log=False, sqrt=False,
        exclude=False, s=False):
    '''
    A routine to generate composite features from a combination of user-provided
    input features.

    developer note: This is currently scales *quite slowly* with max_den.
    There's surely a better way to do this, but it's apparently currently
    functional

    Inputs)
        p: list
            User-provided list of physical features to be combined
        max_num: non-negative integer
            The maximum order of the polynomial in the numerator of the
            composite features
        max_den: non-negative integer
            The maximum order of the polynomial in the denominator of the
            composite features
        log: boolean
            (not currently supported)
            Set to True to include terms involving the logarithm of the input
            features
        sqrt: boolean
            (not currently supported)
            Set to True to include terms involving the square root of the input
            features
        exclude: bool
            Set exclude=True to avoid returning 1 to represent the
            zeroth power
        s: bool
            Set s=True to return a list of strings
            Set s=False to evaluate each element in the list

    Outputs)
        features: list
            A list of combinations of the input features to meet the
            required specifications
    '''
    if max_den == 0:
        return generate_positive_features(p,max_num,exclude=exclude,s=s)
    if max_num == 0:
        dup_feature_keys = generate_positive_features(p,max_den,exclude=exclude,s=True)
        features = []
        for key in dup_feature_keys:
            val = '1/('+key+')'
            features.append(val)
        if not s:
            features = [eval('1.*'+i) for i in features]
        return features
    else:
        num_p = len(p)
        p_str = [str(i) for i in range(num_p)]
        features = []
        feature_keys = generate_positive_features(p_str,max_num,exclude=True,s=True)
        dup_feature_keys = generate_positive_features(p_str,max_den,exclude=True,s=True)
        for key1 in feature_keys:
            l1 = key1.split('*')
            for key2 in dup_feature_keys:
                    l2 = key2.split('*')
                    intersect = list(set.intersection(set(l1),set(l2)))
                    if not intersect:
                        val = _decode_key(p,key1)+'/('+_decode_key(p,key2)+')'
                        features.append(val)
        for key1 in feature_keys:
            features.append(_decode_key(p,key1)+'/(1)')
        for key2 in dup_feature_keys:
            features.append('1/('+_decode_key(p,key2)+')')
        if not exclude:
            features.append('1')
        if not s:
            features = [eval('1.*'+str.replace(i,'^','**')) for i in features]
        return features

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
