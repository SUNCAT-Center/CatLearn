"""Some useful utilities."""
import numpy as np
import hashlib
import json
import time
from atoml.regression import GaussianProcess
import copy

def simple_learning_curve(trainx, trainy, testx, testy,
                          kernel_dict, regularization,
                          step=1, min_data=2,
                          optimize_interval=None, eval_jac=False):
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
    # If the targets are standardized, convert back to the raw targets.
    rmse = []
    mae = []
    signed_mean = []
    Ndata = []
    opt_time = []
    pred_time = []
    lml = []
    gp = GaussianProcess(train_fp=copy.deepcopy(trainx),
                         train_target=copy.deepcopy(trainy),
                         kernel_dict=kernel_dict.copy(),
                         regularization=float(regularization),
                         optimize_hyperparameters=False)
    for low in range(min_data, len(trainx) + 1, step)[::-1]:
        # Update the training data in the gp.
        print('reset.')
        print(kernel_dict, regularization)
        print('update.')
        gp.update_gp(train_fp=copy.deepcopy(trainx[:low, :]),
                     train_target=copy.deepcopy(trainy[:low]),
                     kernel_dict=copy.deepcopy(kernel_dict))
                     # regularization=copy.deepcopy(regularization))
        gp.regularization = float(regularization)
        print(gp.kernel_dict, gp.regularization)
        start = time.time()
        if optimize_interval is not None and low % optimize_interval == 0:
            gp.optimize_hyperparameters(eval_jac=eval_jac)
        end_opt = time.time()
        print(end_opt - start, 'seconds')
        # Do the prediction
        print(np.mean(trainy), np.std(trainy))
        pred = gp.predict(test_fp=copy.deepcopy(testx),
                          get_validation_error=True,
                          get_training_error=False,
                          uncertainty=True,
                          test_target=copy.deepcopy(testy))
        # Store the error associated with the predictions in lists.
        end_pred = time.time()
        Ndata.append(len(trainy[:low]))
        rmse.append(pred['validation_error']['rmse_average'])
        mae.append(pred['validation_error']['absolute_average'])
        signed_mean.append(pred['validation_error']['signed_mean'])
        opt_time.append(end_opt - start)
        pred_time.append(end_pred - end_opt)
        lml.append(gp.log_marginal_likelihood)
    output = {'N_data': Ndata,
              'rmse_average': rmse,
              'absolute_average': mae,
              'signed_mean': signed_mean,
              'opt_time': opt_time,
              'pred_time': pred_time,
              'log_marginal_likelihood': lml}
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
