""" Data generation functions. """
from __future__ import absolute_import
from __future__ import division

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
