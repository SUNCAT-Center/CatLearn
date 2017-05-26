""" Some useful utilities. """
import numpy as np
from collections import defaultdict

from .output import write_data_setup


def remove_outliers(candidates, key, con=1.4826, dev=3., constraint=None,
                    writeout=False):
    """ Preprocessing routine to remove outliers in the data based on the
        median absolute deviation. Only candidates that are unfit, e.g. less
        positive raw_score, are removed as outliers.

        Parameters
        ----------
        con : float
            Constant scale factor dependent on the distribution. Default is
            1.4826 expecting the data is normally distributed.
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


def clean_variance(train, test=None):
    """ Function to remove features that contribute nothing to the model in the
        form of zero variance features.

        Parameters
        ----------
        train : array
            Feature matrix for the traing data.
        test : array
            Feature matrix for the test data.
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
    clean['train'] = train
    clean['test'] = test

    return clean
