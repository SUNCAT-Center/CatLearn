"""Some useful utilities."""
import numpy as np
import hashlib
from scipy.stats import pearsonr, spearmanr, kendalltau
from catlearn.preprocess.scaling import standardize


def formal_charges(atoms, ion_number=8, ion_charge=-2):
    """Return a list of formal charges on atoms.

    Parameters
    ----------
    atoms : object
        ase.Atoms object representing a chalcogenide. The default parameters
        are relevant for an oxide.
    anion_number : int
        atomic number of anion.
    anion_charge : int
        formal charge of anion.

    Returns
    ----------
    all_charges : list
        Formal charges ordered by atomic index.
    """
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
    return all_charges


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
