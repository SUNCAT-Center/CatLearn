"""Some useful utilities."""
import numpy as np
import hashlib


def simple_learning_curve(gp, testx, testy, step=1, min_data=2,
                          optimize_interval=None):
    """Evaluate validation error versus training data size.

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

    Returns
    -------
    N_data : list
        Training data size.
    rmse_average : list
        Root mean square validation error.
    absolute_average : list
        Mean absolute validation error.
    signed_mean : list
        Signed mean validation error.
    """
    if min_data < 2:
        raise ValueError("min_data must be at least 2.")
    # Retrieve the full training data and training targets from gp.
    trainx = (gp.train_fp).copy()
    # If the targets are standardized, convert back to the raw targets.
    trainy = (gp.train_target).copy() * \
        gp.scaling['std'] + \
        gp.scaling['mean']
    rmse = []
    mae = []
    signed_mean = []
    Ndata = []
    for low in range(min_data, len(trainx) + 1, step):
        # Update the training data in the gp. Targets are standardized again.
        gp.update_data(train_fp=trainx[:low, :],
                       train_target=trainy[:low],
                       standardize_target=True, normalize_target=False)
        if optimize_interval is not None and low % optimize_interval == 0:
            gp._optimize_hyperparameters()
        # Do the prediction
        pred = gp.predict(test_fp=testx,
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
    """A hash based strictly on the geometry features of an atoms object.

    Uses positions, cell, and symbols.

    This is intended for planewave basis set calculations, so pbc is not
    considered.

    Each element is sorted in the algorithem to help prevent new hashs for
    identical geometries.
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
