""" Data generation functions. """
import numpy as np
from random import shuffle
from collections import defaultdict


def get_unique(candidates, testsize):
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
            dataset['taken'].append(can[0])
            dataset['candidates'].append(can[1])
        else:
            break

    return dataset


def get_train(candidates, key, trainsize=None, taken_cand=None):
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

    return dataset


def data_split(candidates, nsplit, key):
    """ Routine to split list of candidates into sublists. This can be
        useful for bootstrapping, LOOCV, etc.

        nsplit: int
            The number of bins that data should be devided into.
    """
    dataset = defaultdict(list)
    shuffle(candidates)
    # Calculate the number of items per split.
    n = len(candidates) / nsplit
    # Get any remainders.
    r = len(candidates) % nsplit
    # Define the start and finish of first split.
    s1 = 0
    s2 = n + min(1, r)
    for _ in range(nsplit):
        dataset['split_cand'].append(candidates[int(s1):int(s2)])
        dataset['target'].append([c.info['key_value_pairs'][key]
                                 for c in candidates[int(s1):int(s2)]])
        # Get any new remainder.
        r = max(0, r-1)
        # Define next split.
        s1 = s2
        s2 = s2 + n + min(1, r)

    return dataset


def remove_outliers(candidates, key, con=1.4826, dev=3.):
    """ Preprocessing routine to remove outliers in the data based on the
        median absolute deviation. Only candidates that are unfit, e.g. less
        positive raw_score, are removed as outliers.

        con: float
            Constant scale factor dependent on the distribution. Default is
            1.4826 expecting the data is normally distributed.

        dev: float
            The number of deviations from the median to account for.
    """
    target = []
    for c in candidates:
        target.append(c.info['key_value_pairs'][key])

    vals = np.ma.array(target).compressed()
    # get median
    med = np.median(vals)
    # get median absolute deviation
    mad = con*(np.median(np.abs(vals - med)))

    processed = []
    for c in candidates:
        if c.info['key_value_pairs'][key] > med - (dev * mad):
            processed.append(c)

    return processed


def target_standardize(target):
    """ Returns a list of standardized target values.

        target: list
            A list of the target values.
    """
    target = np.asarray(target)

    data = defaultdict(list)
    data['mean'] = float(np.mean(target))
    data['std'] = float(np.std(target))
    for i in target:
        data['target'].append((i - data['mean']) / data['std'])

    return data
