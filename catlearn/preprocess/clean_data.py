"""Functions to clean data."""
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import Imputer


def remove_outliers(features, targets, con=1.4826, dev=3., constraint=None):
    """Preprocessing routine to remove outliers by median absolute deviation.

    This will take the training feature and target arrays, calculate any
    outliers, then return the reduced arrays. It is possible to set a
    constraint key ('high', 'low', None) in order to allow for outliers that
    are e.g. very low in energy, as this may be the desired outcome of the
    study.

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
    data['mad'] = con * (np.median(np.abs(targets - med)))

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


def clean_variance(train, test=None, labels=None, mask=None):
    """Remove features that contribute nothing to the model.

    Removes a feature if there is zero variance in the training data. If this
    is the case, then the model won't learn anything new from adding this
    feature as it will just act as a scalar.

    Parameters
    ----------
    train : array
        Feature matrix for the traing data.
    test : array
        Optional feature matrix for the test data. Default is None passed.
    labels : array
        Optional list of feature labels. Default is None passed.
    mask : list
        Indices of features that are not subject to cleaning.
    """
    train = np.asarray(train, dtype=np.float64)

    clean = defaultdict(list)

    standard_dev = np.nanstd(train, axis=0)
    assert np.isfinite(standard_dev).all()

    # Index of informative features.
    index = list(np.where(~np.isclose(0, standard_dev))[0])
    clean['index'] = index

    # Clean data.
    clean['train'] = train[:, index].copy()
    if test is not None:
        test = np.asarray(test, dtype=np.float64)
        clean['test'] = test[:, index].copy()
    if labels is not None:
        labels = np.asarray(labels)
        clean['labels'] = labels[index].copy()

    return clean


def clean_infinite(train, test=None, targets=None, labels=None, mask=None,
                   max_impute_fraction=0, strategy='mean'):
    """Remove features that have non finite values in the training data.

    Optionally removes features in test data with non fininte values. Returns
    a dictionary with the clean 'train', 'test' and 'index' that were removed
    from the original data.

    Parameters
    ----------
    train : array
        Feature matrix for the traing data.
    test : array
        Optional feature matrix for the test data. Default is None passed.
    targets : array
        An array of training targets.
    labels : array
        Optional list of feature labels. Default is None passed.
    mask : list
        Indices of features that are not subject to cleaning.
    max_impute_fraction : float
        Maximum fraction of values in a column that can be imputed.
        Columns with higher fractions of nans values will be discarded.
    strategy : str
        Imputation strategy.

    Returns
    --------
    data : dict
        key value pairs

            - 'train' : array
                Clean training data matrix.
            - 'test' : array
                Clean test data matrix
            - 'targets' : list
                Boolean list on whether targets are finite.
            - 'labels' : list
                Feature labels of clean data set.

    """
    clean = defaultdict(list)

    train = np.array(train, dtype=np.float64)

    if targets is not None:
        targets = np.reshape(targets, [len(targets), 1])
        bool_test = np.isfinite(targets).all(axis=1)
        clean['targets'] = targets[bool_test]
        train = train[bool_test, :]

    if test is not None:
        test = np.asarray(test, dtype=np.float64)

    # Get the fraction of finite values in each column.
    if max_impute_fraction > 0:
        impute = Imputer(missing_values="NaN", strategy=strategy)
        impute_fraction = 1 - np.isfinite(train).mean(axis=0)
        to_impute = impute_fraction <= max_impute_fraction
        train[:, to_impute] = impute.fit_transform(train[:, to_impute])
        if test is not None:
            test[:, to_impute] = impute.transform(test[:, to_impute])

    # Find features that have only finite values.
    bool_test = np.isfinite(train).all(axis=0)
    # Save the indices of columns that contain only finite values.
    clean['index'] = list(np.where(bool_test)[0])

    # Also accept features, that are masked.
    if mask is not None:
        bool_test[mask] = True

    # Save a cleaned training data matrix.
    clean['train'] = train[:, clean['index']]
    # If a test matrix is given, save a cleaned test data matrix.
    if test is not None:
        assert int(np.shape(test)[1]) == int(np.shape(train)[1])
        test = test[:, clean['index']]
    clean['test'] = test
    if labels is not None:
        assert len(labels) == int(np.shape(train)[1])
        labels = list(np.array(labels)[clean['index']])
    clean['labels'] = labels

    return clean
