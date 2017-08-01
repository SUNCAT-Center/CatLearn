"""Functions to process the raw feature matrix."""
import numpy as np
from scipy import cluster
from collections import defaultdict

from .output import write_fingerprint_setup


def matrix_split(X, nsplit, fix_size=None):
    """Routine to split feature matrix and return sublists.

    Parameters
    ----------
    nsplit : int
        The number of bins that data should be devided into.
    fix_size : int
        Define a fixed sample size, e.g. nsplit=5 fix_size=100, generates
        5 x 100 data split. Default is None, all avaliable data is divided
        nsplit times.
    """
    np.random.shuffle(X)  # Shuffle ordering of the array along 0 axis.
    if fix_size is not None:
        msg = 'Cannot divide dataset in this way, number of candidates is '
        msg += 'too small'
        assert np.shape(X)[0] >= nsplit * fix_size, msg

        X = X[:nsplit * fix_size, :]

    return np.array_split(X, nsplit)


def standardize(train_matrix, test_matrix=None, mean=None, std=None,
                writeout=False):
    """Standardize each feature relative to the mean and standard deviation.

    If test data is supplied it is standardized relative to the training
    dataset.

    Parameters
    ----------
    train_matrix : list
        Feature matrix for the training dataset.
    test_matrix : list
        Feature matrix for the test dataset.
    mean : list
        List of mean values for each feature.
    std : list
        List of standard deviation values for each feature.
    """
    scale = defaultdict(list)
    if mean is None:
        mean = np.mean(train_matrix, axis=0)
    scale['mean'] = mean
    if std is None:
        std = np.std(train_matrix, axis=0)
    scale['std'] = std
    np.place(scale['std'], scale['std'] == 0., [1.])  # Replace 0 with 1.

    scale['train'] = (train_matrix - scale['mean']) / scale['std']

    if test_matrix is not None:
        test_matrix = (test_matrix - scale['mean']) / scale['std']
    scale['test'] = test_matrix

    if writeout:
        write_fingerprint_setup(function='standardize', data=scale)

    return scale


def normalize(train_matrix, test_matrix=None, mean=None, dif=None,
              writeout=False):
    """Normalize each feature relative to mean and min/max variance.

    If test data is supplied it is standardized relative to the training
    dataset.

    Parameters
    ----------
    train_matrix : list
        Feature matrix for the training dataset.
    test_matrix : list
        Feature matrix for the test dataset.
    mean : list
        List of mean values for each feature.
    dif : list
        List of max-min values for each feature.
    """
    scale = defaultdict(list)
    if mean is None:
        mean = np.mean(train_matrix, axis=0)
    scale['mean'] = mean
    if dif is None:
        dif = np.max(train_matrix, axis=0) - np.min(train_matrix, axis=0)
    scale['dif'] = dif
    np.place(scale['dif'], scale['dif'] == 0., [1.])  # Replace 0 with 1.

    scale['train'] = (train_matrix - scale['mean']) / scale['dif']

    if test_matrix is not None:
        test_matrix = (test_matrix - scale['mean']) / scale['dif']
    scale['test'] = test_matrix

    if writeout:
        write_fingerprint_setup(function='normalize', data=scale)

    return scale


def cluster_features(train_matrix, train_target, k=2, test_matrix=None,
                     test_target=None):
    """Function to perform k-means clustering in the feature space.

    Parameters
    ----------
    train_matrix : list
        Feature matrix for the training dataset.
    train_target : list
        List of target values for training data.
    k : int
        Number of clusters to divide data into.
    test_matrix : list
        Feature matrix for the test dataset.
    test_target : list
        List of target values for test data.
    """
    m = defaultdict(list)

    centroids, order = cluster.vq.kmeans2(train_matrix, k)

    # Break up the training data based on clusters.
    train_data = _cluster_split(feature_matrix=train_matrix,
                                target=train_target, order=order)
    m['train_features'], m['train_target'] = train_data[0], train_data[1]

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
    test_data = _cluster_split(feature_matrix=test_matrix, target=test_target,
                               order=m['test_order'])
    m['test_features'], m['test_target'] = test_data[0], test_data[1]

    return m


def _cluster_split(feature_matrix, target, order):
    """Function to split up data based on clustering."""
    split_f = defaultdict(list)
    if target is not None:
        split_t = defaultdict(list)
        for f, t, l in zip(feature_matrix, target, order):
            split_f[l].append(f)
            split_t[l].append(t)
    else:
        split_t = None
        for f, t, l in zip(feature_matrix, order):
            split_f[l].append(f)

    return split_f, split_t
