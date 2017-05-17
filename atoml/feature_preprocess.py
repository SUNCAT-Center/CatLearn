""" Functions to process the raw feature matrix. """
import numpy as np
from scipy import cluster
from random import shuffle
from collections import defaultdict

from .output import write_fingerprint_setup


def matrix_split(X, nsplit, fix_size=None, replacement=False):
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
            s2 += n + min(1, r)
        else:
            s2 += fix_size
    return dataset


def standardize(train_matrix, test_matrix=None, writeout=False):
    """ Standardize each feature relative to the mean and standard deviation.
        If test data is supplied it is standardized relative to the training
        dataset.

        Parameters
        ----------
        train_matrix : list
            Feature matrix for the training dataset.
        test_matrix : list
            Feature matrix for the test dataset.
    """
    feature_mean = np.mean(train_matrix, axis=0)
    feature_std = np.std(train_matrix, axis=0)
    # Replace zero std with value 1 for devision.
    np.place(feature_std, feature_std == 0., [1.])

    std = defaultdict(list)
    std['train'] = (train_matrix - feature_mean) / feature_std
    if test_matrix is not None:
        test_matrix = (test_matrix - feature_mean) / feature_std

    std['test'] = test_matrix
    std['mean'] = feature_mean
    std['std'] = feature_std

    if writeout:
        write_fingerprint_setup(function='standardize', data=std)

    return std


def normalize(train_matrix, test_matrix=None, writeout=False):
    """ Normalize each feature relative to mean and min/max variance. If
        test data is supplied it is standardized relative to the training
        dataset.

        Parameters
        ----------
        train_matrix : list
            Feature matrix for the training dataset.
        test_matrix : list
            Feature matrix for the test dataset.
    """
    feature_mean = np.mean(train_matrix, axis=0)
    feature_dif = np.max(train_matrix, axis=0) - np.min(train_matrix, axis=0)
    # Replace zero difference with value 1 for devision.
    np.place(feature_dif, feature_dif == 0., [1.])

    norm = defaultdict(list)
    norm['train'] = (train_matrix - feature_mean) / feature_dif
    if test_matrix is not None:
        test_matrix = (test_matrix - feature_mean) / feature_dif

    norm['test'] = test_matrix
    norm['mean'] = feature_mean
    norm['dif'] = feature_dif

    if writeout:
        write_fingerprint_setup(function='normalize', data=norm)

    return norm


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
