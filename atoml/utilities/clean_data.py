"""Functions to clean data."""
import numpy as np
from collections import defaultdict


def remove_outliers(features, targets, con=1.4826, dev=3., constraint=None):
    """Preprocessing routine to remove outliers by median absolute deviation.

    Parameters
    ----------
    features : array
        Feature matrix for training data.
    targets : list
        List of target values for the training data.
    con : float
        Constant scale factor dependent on the distribution. Default is 1.4826
        expecting the data is normally distributed.
    dev : float
        The number of deviations from the median to account for.
    constraint : str
        Can be set to 'low' to remove candidates with targets that are too
        small/negative or 'high' for outliers that are too large/positive.
        Default is to remove all.
    """
    data = defaultdict(list)
    # get median
    med = np.median(targets)
    # get median absolute deviation
    data['mad'] = con*(np.median(np.abs(targets - med)))

    if constraint is 'high' or constraint is None:
        m = np.ma.masked_less(targets, med + data['mad'])
        targets = targets[m.mask]
        features = features[m.mask]

    if constraint is 'low' or constraint is None:
        m = np.ma.masked_greater(targets, med - data['mad'])
        targets = targets[m.mask]
        features = features[m.mask]

    data['targets'], data['features'] = targets, features

    return data


def clean_variance(train, test=None, labels=None):
    """Remove features that contribute nothing to the model.

    Parameters
    ----------
    train : array
        Feature matrix for the traing data.
    test : array
        Optional feature matrix for the test data.
    """
    clean = defaultdict(list)
    m = train.T
    # Find features that provide no input for model.
    for i in list(range(len(m))):
        if np.allclose(m[i], m[i][0]):
            clean['index'].append(i)
    # Remove bad data from feature matrix.
    if 'index' in clean:
        train = np.delete(m, clean['index'], axis=0).T
        if test is not None:
            test = np.delete(test.T, clean['index'], axis=0).T
        if labels is not None:
            labels = np.delete(labels, clean['index'])
    clean['train'] = train
    clean['test'] = test
    clean['labels'] = labels

    return clean


def clean_infinite(train, test=None, labels=None):
    """Remove features that have non finite values in the training data.

    Optionally removes features in test data with non fininte values. Returns
    a dictionary with the clean 'train', 'test' and 'index' that were removed
    from the original data.

    Parameters
    ----------
    train : array
        Feature matrix for the traing data.
    test : array
        Optional feature matrix for the test data.
    """
    clean = defaultdict(list)
    # Find features that have only finite values.
    bool_test = np.isfinite(train).all(axis=0)
    # Save the indices of columns that contain non-finite values.
    clean['index'] = list(np.where(~bool_test)[0])
    # Save a cleaned training data matrix.
    clean['train'] = train[:, bool_test]
    # If a test matrix is given, save a cleaned test data matrix.
    if test is not None:
        assert int(np.shape(test)[1]) == int(np.shape(train)[1])
        test = test[:, bool_test]
    clean['test'] = test
    if labels is not None:
        assert len(labels) == int(np.shape(train)[1])
        labels = list(np.array(labels)[bool_test])
    clean['labels'] = labels

    return clean
