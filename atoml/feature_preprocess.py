import numpy as np
from random import shuffle
from collections import defaultdict

from .output import write_fingerprint_setup


def fpmatrix_split(X, nsplit, fix_size=None, replacement=False):
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
            s1 += n + min(1, r)
        else:
            s1 += fix_size
    return dataset


def standardize(train, test=None, writeout=False):
    """ Standardize each descriptor in the FPV relative to the mean and
        standard deviation. If test data is supplied it is standardized
        relative to the training dataset.

        train: list
            List of atoms objects to be used as training dataset.

        test: list
            List of atoms objects to be used as test dataset.
    """
    mean_fpv = np.mean(train, axis=0)
    std_fpv = np.std(train, axis=0)
    # Replace zero std with value 1 for devision.
    np.place(std_fpv, std_fpv == 0., [1.])

    std = defaultdict(list)
    std['train'] = (train - mean_fpv) / std_fpv
    if test is not None:
        test = (test - mean_fpv) / std_fpv

    std['test'] = test
    std['std'] = std_fpv
    std['mean'] = mean_fpv

    if writeout:
        write_fingerprint_setup(function='standardize', data=std)

    return std


def normalize(train, test=None, writeout=False):
    """ Normalize each descriptor in the FPV to min/max or mean centered. If
        test data is supplied it is standardized relative to the training
        dataset.
    """
    mean_fpv = np.mean(train, axis=0)
    dif = np.max(train, axis=0) - np.min(train, axis=0)
    # Replace zero difference with value 1 for devision.
    np.place(dif, dif == 0., [1.])

    norm = defaultdict(list)
    norm['train'] = (train - mean_fpv) / dif
    if test is not None:
        test = (test - mean_fpv) / dif

    norm['test'] = test
    norm['mean'] = mean_fpv
    norm['dif'] = dif

    if writeout:
        write_fingerprint_setup(function='normalize', data=norm)

    return norm
