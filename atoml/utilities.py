"""Some useful utilities."""
import numpy as np
from collections import defaultdict


def remove_outliers(candidates, key, con=1.4826, dev=3., constraint=None):
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

    Optionally removes features in test data with non fininte values. Returns
    a dictionary with the clean 'train', 'test' and 'index' that were removed
    from the original data.

    Parameters
    ----------
    train : array
        Feature matrix for the traing data.
    test : array
        Optional feature matrix for the test data.
    """
    clean = defaultdict(list)
    # Find features that have only finite values.
    bool_test = np.isfinite(train).all(axis=0)
    # Save the indices of columns that contain non-finite values.
    clean['index'] = list(np.where(~bool_test)[0])
    # Save a cleaned training data matrix.
    clean['train'] = train[:, bool_test]
    # If a test matrix is given, save a cleaned test data matrix.
    if test is not None:
        assert int(np.shape(test)[1]) == int(np.shape(train)[1])
        test = test[:, bool_test]
    clean['test'] = test
    if labels is not None:
        assert len(labels) == int(np.shape(train)[1])
        labels = list(np.array(labels)[bool_test])
    clean['labels'] = labels

    return clean


def test_data_limited(gp, testx, testy, step=1, min_data=2,
                      optimize_interval=None):
    """Evaluate validation error versus training data size.

    Returns a dictionary containing the lists:
    -------
    N_data : training data size
    rmse_average : Root mean square validation error.
    absolute_average : mean absolute validation error.
    signed_mean : signed mean validation error.

    Parameters
    ----------
    gp : GaussianProcess class
        create it from atoml.GaussianProcess
    testx : array
        Feature matrix for the test data.
    testy : list
        A list of the the test targets used to generate the prediction
        errors.
    step : integer
        Number of datapoints per prediction iteration.
    min_data : integer
        Number of datapoints in first prediction iteration.
    optimize_interval : integer or None
        Reoptimize the hyperparameters every this many datapoints.
    """
    if min_data < 2:
        raise ValueError("min_data must be at least 2.")
    # Retrieve the full training data and training targets from gp.
    trainx = (gp.train_fp).copy()
    # If the targets are standardized, convert back to the raw targets.
    trainy = (gp.train_target).copy() * \
        gp.standardize_data['std'] + \
        gp.standardize_data['mean']
    rmse = []
    mae = []
    signed_mean = []
    Ndata = []
    for low in range(min_data, len(trainx)+1, step):
        # Update the training data in the gp. Targets are standardized again.
        gp.update_data(train_fp=trainx[:low, :],
                       train_target=trainy[:low],
                       standardize_target=True)
        if optimize_interval is not None and low % optimize_interval == 0:
            gp.optimize_hyperparameters()
        # Do the prediction
        pred = gp.get_predictions(test_fp=testx,
                                  get_validation_error=True,
                                  get_training_error=False,
                                  uncertainty=True,
                                  test_target=testy)
        # Store the error associated with the predictions in lists.
        Ndata.append(len(trainy[:low]))
        rmse.append(pred['validation_error']['rmse_average'])
        mae.append(pred['validation_error']['absolute_average'])
        signed_mean.append(pred['validation_error']['signed_mean'])
    output = {'N_data': Ndata,
              'rmse_average': rmse,
              'absolute_average': mae,
              'signed_mean': signed_mean}
    return output


def geometry_hash(atoms):
    """ A hash based strictly on the geometry features of
    an atoms object: positions, cell, and symbols.

    This is intended for planewave basis set calculations,
    so pbc is not considered.

    Each element is sorted in the algorithem to help prevent
    new hashs for identical geometries.
    """

    atoms.wrap()

    pos = atoms.get_positions()

    # Sort the cell array by magnitude of z, y, x coordinates, in that order
    cell = np.array(sorted(atoms.get_cell(),
                           key=lambda x: (x[2], x[1], x[0])))

    # Flatten the array and return a string of numbers only
    # We only consider position changes up to 3 decimal places
    cell_hash = np.array_str(np.ndarray.flatten(cell.round(3)))
    cell_hash = ''.join(cell_hash.strip('[]').split()).replace('.', '')

    # Sort the atoms positions similarly, but store the sorting order
    pos = atoms.get_positions()
    srt = [i for i, _ in sorted(enumerate(pos),
                                key=lambda x: (x[1][2], x[1][1], x[1][0]))]
    pos_hash = np.array_str(np.ndarray.flatten(pos[srt].round(3)))
    pos_hash = ''.join(pos_hash.strip('[]').split()).replace('.', '')

    # Create a symbols hash in the same fashion conserving position sort order
    sym = np.array(atoms.get_atomic_numbers())[srt]
    sym_hash = np.array_str(np.ndarray.flatten(sym))
    sym_hash = ''.join(sym_hash.strip('[]').split())

    # Assemble a master hash and convert it through an md5
    master_hash = cell_hash + pos_hash + sym_hash
    md5 = hashlib.md5(master_hash)
    _hash = md5.hexdigest()

    return _hash
