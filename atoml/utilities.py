"""Some useful utilities."""
import numpy as np
from collections import defaultdict

from .output import write_data_setup


def remove_outliers(candidates, key, con=1.4826, dev=3., constraint=None,
                    writeout=False):
    """Preprocessing routine to remove outliers by median absolute deviation.

    Parameters
    ----------
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
    dataset = defaultdict(list)
    target = []
    order = list(range(len(candidates)))
    for c in candidates:
        target.append(c.info['key_value_pairs'][key])

    vals = np.ma.array(target).compressed()
    # get median
    med = np.median(vals)
    # get median absolute deviation
    mad = con*(np.median(np.abs(vals - med)))

    if constraint is None:
        for c, o in zip(candidates, order):
            if c.info['key_value_pairs'][key] > med - (dev * mad) and \
               c.info['key_value_pairs'][key] < med + (dev * mad):
                dataset['processed'].append(c)
            else:
                dataset['removed'].append(o)

    elif constraint is 'low':
        for c, o in zip(candidates, order):
            if c.info['key_value_pairs'][key] > med - (dev * mad):
                dataset['processed'].append(c)
            else:
                dataset['removed'].append(o)

    elif constraint is 'high':
        for c, o in zip(candidates, order):
            if c.info['key_value_pairs'][key] < med + (dev * mad):
                dataset['processed'].append(c)
            else:
                dataset['removed'].append(o)

    if writeout:
        write_data_setup(function='remove_outliers', data=dataset)

    return dataset


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
        Optionally removes datapoints in test data with non fininte values.
        Returns a dictionary with the clean 'train', 'test' and 'index' that
            were removed from the original data.

    Parameters
    ----------
    train : array
        Feature matrix for the traing data.
    test : array
        Optional feature matrix for the test data.
    """
    clean = defaultdict(list)
    # Find features that have only finite values.
    l = np.isfinite(train).all(axis=0)
    # Save the indices of columns that contain non-finite values.
    clean['index'] = list(np.where(~l)[0])
    # Save a cleaned training data matrix.
    clean['train'] = train[:,l]
    # If a test matrix is given, save a cleaned test data matrix.
    if test is not None:
        assert int(np.shape(test)[1]) == int(np.shape(train)[1])
        clean['test'] = test[:,l]
    else:
        clean['test'] = test
    if labels is not None:
        assert len(labels) == int(np.shape(train)[1])
        clean['labels'] = np.array(labels)[l]
    else:
        clean['labels'] = list(labels)

    return clean
