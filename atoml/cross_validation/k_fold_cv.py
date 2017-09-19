"""Setup k-fold array split for cross validation."""
import numpy as np


def k_fold(X, nsplit, fix_size=None):
    """Routine to split feature matrix and return sublists.

    Parameters
    ----------
    nsplit : int
        The number of bins that data should be devided into.
    fix_size : int
        Define a fixed sample size, e.g. nsplit=5 fix_size=100, generates
        5 x 100 data split. Default is None, all available data is divided
        nsplit times.
    """
    np.random.shuffle(X)  # Shuffle ordering of the array along 0 axis.
    if fix_size is not None:
        msg = 'Cannot divide dataset in this way, number of candidates is '
        msg += 'too small'
        assert np.shape(X)[0] >= nsplit * fix_size, msg

        X = X[:nsplit * fix_size, :]

    return np.array_split(X, nsplit)
