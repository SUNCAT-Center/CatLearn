"""Some useful utilities."""
import numpy as np
import hashlib
import time
import multiprocessing
from tqdm import trange, tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
from catlearn.preprocess.scaling import standardize


def formal_charges(atoms, ion_number, ion_charge):
    cm = atoms.connectivity
    anion_charges = np.zeros(len(atoms))
    for i, atom in enumerate(atoms):
        if atoms.numbers[i] == ion_number:
            anion_charges[i] = ion_charge
            transfer = cm * np.vstack(anion_charges)
            row_sums = transfer.sum(axis=1)
            for j, s in enumerate(row_sums):
                if s == ion_charge:
                    row_sums[j] *= abs(ion_charge)
            shared = ion_charge * transfer / np.vstack(row_sums)
            cation_charges = -np.nansum(shared, axis=0)
            all_charges = anion_charges + cation_charges
    atoms.set_initial_charges(all_charges)
    return atoms


def holdout_set(data, fraction, target=None, seed=None):
    """Return a dataset split in a hold out set and a training set.

    Parameters
    ----------
    matrix : array
        n by d array
    fraction : float
        fraction of data to hold out for testing.
    target : list
        optional list of targets or separate feature.
    seed : float
        optional float for reproducible splits.
    """
    matrix = np.array(data)

    # Randomize order.
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(matrix)

    # Split data.
    index = int(len(matrix) * fraction)
    holdout = matrix[:index, :]
    train = matrix[index:, :]

    if target is None:
        return train, holdout

    train_target = target[:index]
    test_target = target[index:]

    return train, train_target, holdout, test_target


def target_correlation(train, target,
                       correlation=['pearson', 'spearman', 'kendall']):
    """Return the correlation of all columns of train with a target feature.

    Parameters
    ----------
    train : array
        n by d training data matrix.
    target : list
        target for correlation.

    Returns
    -------
    metric : array
        len(metric) by d matrix of correlation coefficients.
    """
    # Scale and shape the data.
    train_data = standardize(train_matrix=train)['train']
    train_target = target
    output = []
    for c in correlation:
        correlation = c
        # Find the correlation.
        row = []
        for d in train_data.T:
            if correlation is 'pearson':
                row.append(pearsonr(d, train_target)[0])
            elif correlation is 'spearman':
                row.append(spearmanr(d, train_target)[0])
            elif correlation is 'kendall':
                row.append(kendalltau(d, train_target)[0])
        output.append(row)

    return output


class LearningCurve(object):
    """The simple learning curve class."""

    def __init__(self, nprocs=1):
        """Initialize the class.

        Parameters
        ----------
        nprocs : int
            Number of processers used in parallel implementation. Default is 1
            e.g. serial.
        """
        self.nprocs = nprocs

    def learning_curve(self, predict, train, target, test, test_target,
                       step=1, min_data=2):
        """Evaluate custom metrics versus training data size.

        Parameters
        ----------
        predict : object
            A function that will make the predictions. predict should accept
            the parameters:

                train_features : array
                test_features : array
                train_targets : list
                test_targets : list

            predict should return either a float or a list of floats. The float
            or the first value of the list will be used as the fitness score.
        train : array
            An n, d array of training examples.
        targets : list
            A list of the target values.
        test : array
            An n, d array of test data.
        test targets : list
            A list of the test target values.
        step : int
            Incrementent the data set size by this many examples.
        min_data : int
            Smallest number of training examples to test.

        Returns
        -------
        output : array
            Each row is the output from the predict object.
        """
        n, d = np.shape(train)
        # Get total number of iterations
        total = (n - min_data) // step
        output = []
        # Iterate through the data subset.
        if self.nprocs != 1:
            # First a parallel implementation.
            pool = multiprocessing.Pool(self.nprocs)
            tasks = np.arange(total)
            args = (
                (x, step, train, test, target,
                 test_target, predict) for x in tasks)
            for r in tqdm(pool.imap_unordered(
                    _single_model, args), total=total,
                    desc='nested              ', leave=False):
                output.append(r)
                # Wait to make things more stable.
                time.sleep(0.001)
            pool.close()
        else:
            # Then a more clear serial implementation.
            for x in trange(
                    total,
                    desc='nested              ', leave=False):
                args = (x, step, train, test,
                        target, test_target, predict)
                r = _single_model(args)
                output.append(r)
        return output


def _single_model(args):
    """Run a model on a subset of training data with a fixed test set.

    Return the output of a function specified by the last argument.

    Parameters
    ----------
    args : tuple
        Parameters and data to be passed to model.

        args[0] : int
            Increment.
        args[1] : int
            Step size. args[1] * args[0] training examples will be passed
            to the regression model.
        args[2] : array
            An n, d array of training examples.
        args[3] : list
            A list of the target values.
        args[4] : array
            An n, d array of test data.
        args[5] : list
            A list of the test target values.
        args[6] : object
            custom function testing a regression model.
            Must accept 4 parameters, which are args[2:5].
    """
    # Unpack args tuple.
    x = args[0]
    n = x * args[1]
    train_features = args[2]
    test = args[3]
    train_targets = args[4]
    test_targets = args[5]
    predict = args[6]

    # Delete required subset of training examples.
    train = train_features[-n:, :]
    targets = train_targets[-n:]

    # Calculate the error or other metrics from the model.
    result = predict(train, targets, test, test_targets)
    return result


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
