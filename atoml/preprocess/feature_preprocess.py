"""Functions to process the raw feature matrix."""
import numpy as np
from scipy import cluster
from collections import defaultdict


def standardize(train_matrix, test_matrix=None, mean=None, std=None,
                local=True):
    """Standardize each feature relative to the mean and standard deviation.

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
    local : boolean
        Define whether to scale locally or globally.
    """
    scale = defaultdict(list)
    if test_matrix is not None and not local:
        data = np.concatenate((train_matrix, test_matrix), axis=0)
    else:
        data = train_matrix

    if mean is None:
        mean = np.mean(data, axis=0)
    scale['mean'] = mean
    if std is None:
        std = np.std(data, axis=0)
    scale['std'] = std
    np.place(scale['std'], scale['std'] == 0., [1.])  # Replace 0 with 1.

    scale['train'] = (train_matrix - scale['mean']) / scale['std']

    if test_matrix is not None:
        test_matrix = (test_matrix - scale['mean']) / scale['std']
    scale['test'] = test_matrix

    return scale


def normalize(train_matrix, test_matrix=None, mean=None, dif=None, local=True):
    """Normalize each feature relative to mean and min/max variance.

    Parameters
    ----------
    train_matrix : list
        Feature matrix for the training dataset.
    test_matrix : list
        Feature matrix for the test dataset.
    local : boolean
        Define whether to scale locally or globally.
    mean : list
        List of mean values for each feature.
    dif : list
        List of max-min values for each feature.
    """
    scale = defaultdict(list)
    if test_matrix is not None and not local:
        data = np.concatenate((train_matrix, test_matrix), axis=0)
    else:
        data = train_matrix

    if mean is None:
        mean = np.mean(data, axis=0)
    scale['mean'] = mean
    if dif is None:
        dif = np.max(data, axis=0) - np.min(data, axis=0)
    scale['dif'] = dif
    np.place(scale['dif'], scale['dif'] == 0., [1.])  # Replace 0 with 1.

    scale['train'] = (train_matrix - scale['mean']) / scale['dif']

    if test_matrix is not None:
        test_matrix = (test_matrix - scale['mean']) / scale['dif']
    scale['test'] = test_matrix

    return scale


def min_max(train_matrix, test_matrix=None, local=True):
    """Normalize each feature relative to the min and max.

    Parameters
    ----------
    train_matrix : list
        Feature matrix for the training dataset.
    test_matrix : list
        Feature matrix for the test dataset.
    local : boolean
        Define whether to scale locally or globally.
    """
    norm = defaultdict(list)
    if test_matrix is not None and not local:
        data = np.concatenate((train_matrix, test_matrix), axis=0)
    else:
        data = train_matrix
    norm['min'] = np.min(data, axis=0)
    norm['dif'] = np.max(data, axis=0) - norm['min']
    np.place(norm['dif'], norm['dif'] == 0., [1.])  # Replace 0 with 1.

    norm['train'] = (train_matrix - norm['min']) / norm['dif']

    if test_matrix is not None:
        test_matrix = (test_matrix - norm['min']) / norm['dif']
    norm['test'] = test_matrix

    return norm


def unit_length(train_matrix, test_matrix=None, local=True):
    """Normalize each feature relative to the Euclidean length.

    Parameters
    ----------
    train_matrix : list
        Feature matrix for the training dataset.
    test_matrix : list
        Feature matrix for the test dataset.
    local : boolean
        Define whether to scale locally or globally.
    """
    norm = defaultdict(list)
    if test_matrix is not None and not local:
        data = np.concatenate((train_matrix, test_matrix), axis=0)
    else:
        data = train_matrix
    norm['length'] = np.linalg.norm(data, axis=0)
    np.place(norm['length'], norm['length'] == 0., [1.])  # Replace 0 with 1.

    norm['train'] = train_matrix / norm['length']

    if test_matrix is not None:
        test_matrix = test_matrix / norm['length']
    norm['test'] = test_matrix

    return norm


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
