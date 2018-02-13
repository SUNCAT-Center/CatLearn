"""Setup k-fold array split for cross validation."""
import numpy as np


def k_fold(features, targets, nsplit, fix_size=None):
    """Routine to split feature matrix and return sublists.

    Parameters
    ----------
    features : array
        An n, d feature array.
    targets : list
        A list to target values.
    nsplit : int
        The number of bins that data should be devided into.
    fix_size : int
        Define a fixed sample size, e.g. nsplit=5 fix_size=100, generates
        5 x 100 data split. Default is None, all available data is divided
        nsplit times.

    Returns
    -------
    features : list
        A list of feature arrays of length nsplit.
    targets : list
        A list of targets lists of length nsplit.
    """
    # Stick features and targets together.
    d, f = np.shape(features)
    X = np.concatenate(
        (features, np.reshape(targets, (len(targets), 1))), axis=1)
    assert (d, f + 1) == np.shape(X)

    # Shuffle the combined array.
    np.random.shuffle(X)  # Shuffle ordering of the array along 0 axis.

    if fix_size is not None:
        msg = 'Cannot divide dataset in this way, number of candidates is '
        msg += 'too small'
        assert np.shape(X)[0] >= nsplit * fix_size, msg

        X = X[:nsplit * fix_size, :]

    # Split the feature-targets array.
    X = np.array_split(X, nsplit)

    # Split the features and targets, generating two lists.
    features, targets = [], []
    for i in X:
        features.append(i[:, :-1])
        targets.append(i[:, -1])

    return features, targets
