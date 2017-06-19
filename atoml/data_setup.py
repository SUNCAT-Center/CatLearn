"""Data generation functions to interact with ASE atoms objects."""
from __future__ import absolute_import
from __future__ import division

from random import shuffle
from collections import defaultdict

from .output import write_data_setup


def get_unique(atoms, size, key, writeout=False):
    """Return a unique test dataset.

    Parameters
    ----------
    atoms : list
        A list of ASE atoms objects.
    size : int
        Size of unique dataset to be returned.
    key : string
        Property on which to base the predictions stored in the atoms object as
        atoms.info['key_value_pairs'][key].
    """
    dataset = defaultdict(list)
    # Get order of candidates.
    orderlist = list(enumerate(atoms))
    # Shuffle the ordered dataset so returned set is randomly selected.
    shuffle(orderlist)
    # Get data set and record order value to ensure set remains unique.
    for a in orderlist:
        if len(dataset['atoms']) < size:
            dataset['atoms'].append(a[1])
            dataset['target'].append(a[1].info['key_value_pairs'][key])
            dataset['taken'].append(a[0])
        else:
            break

    if writeout:
        write_data_setup(function='get_unique', data=dataset)

    return dataset


def get_train(atoms, key, size=None, taken=None, writeout=False):
    """Return a training dataset.

    Parameters
    ----------
    atoms : list
        A list of ASE atoms objects.
    size : int
        Size of training dataset.
    taken : list
        List of candidates that have been used in unique dataset.
    key : string
        Property on which to base the predictions stored in the atoms object as
        atoms.info['key_value_pairs'][key].
    """
    dataset = defaultdict(list)
    orderlist = list(enumerate(atoms))
    shuffle(orderlist)
    # return candidates not in test set.
    for a in orderlist:
        if size is None or size is not None and len(dataset['atoms']) < size:
            if taken is None or taken is not None and a[0] not in taken:
                dataset['atoms'].append(a[1])
                dataset['target'].append(a[1].info['key_value_pairs'][key])
                dataset['order'].append(a[0])
        elif size is not None and len(dataset['atoms']) == size:
            break

    if writeout:
        write_data_setup(function='get_train', data=dataset)

    return dataset
