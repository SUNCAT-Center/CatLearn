""" Data generation functions. """
from __future__ import absolute_import
from __future__ import division

import numpy as np
from random import shuffle
from collections import defaultdict

from .output import write_data_setup


def get_unique(candidates, testsize, key, writeout=False):
    """ Returns a unique test dataset in the form of a integer list, to track
        selected candidates, and a list of atoms objects making up the set.
    """
    dataset = defaultdict(list)
    # Get order of candidates.
    orderlist = list(zip(range(len(candidates)), candidates))
    # Shuffle the ordered dataset so returned set is randomly selected.
    shuffle(orderlist)
    # Get data set and record order value to ensure set remains unique.
    for can in orderlist:
        if len(dataset['candidates']) < testsize:
            dataset['candidates'].append(can[1])
            dataset['target'].append(can[1].info['key_value_pairs'][key])
            dataset['taken'].append(can[0])
        else:
            break

    if writeout:
        write_data_setup(function='get_unique', data=dataset)

    return dataset


def get_train(candidates, key, trainsize=None, taken_cand=None,
              writeout=False):
    """ Returns a training dataset in the form of a list of atoms objects
        making up the set and a list of the target values. The list is in a
        random order. If the original order is required, use the 'order' list.

        trainsize: int
            Size of training dataset.

        taken_cand: list
            List of candidates that have been used in unique dataset.

        key: string
            Property on which to base the predictions stored in the atoms
            object as atoms.info['key_value_pairs'][key].
    """
    dataset = defaultdict(list)
    orderlist = list(zip(range(len(candidates)), candidates))
    shuffle(orderlist)
    # return candidates not in test set.
    for can in orderlist:
        if trainsize is None or trainsize is not None and \
                len(dataset['candidates']) < trainsize:
            if taken_cand is None or taken_cand is not None and \
                        can[0] not in taken_cand:
                dataset['candidates'].append(can[1])
                dataset['target'].append(can[1].info['key_value_pairs'][key])
                dataset['order'].append(can[0])
        elif trainsize is not None and \
                len(dataset['candidates']) == trainsize:
            break

    if writeout:
        write_data_setup(function='get_train', data=dataset)

    return dataset


def remove_outliers(candidates, key, con=1.4826, dev=3., constraint=None,
                    writeout=False):
    """ Preprocessing routine to remove outliers in the data based on the
        median absolute deviation. Only candidates that are unfit, e.g. less
        positive raw_score, are removed as outliers.

        con: float
            Constant scale factor dependent on the distribution. Default is
            1.4826 expecting the data is normally distributed.

        dev: float
            The number of deviations from the median to account for.

        constraint: str
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
