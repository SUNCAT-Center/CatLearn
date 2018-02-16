"""Functions to setup fingerprint vectors."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from collections import defaultdict
from tqdm import tqdm

def get_combined_descriptors(labels):
    """Sequentially combine feature label vectors.

    Parameters
    ----------
    fpv_list : list
        Functions that return fingerprints.
    """
    # Check that there are at least two fingerprint descriptors to combine.
    msg = "This functions combines various fingerprint"
    msg += " vectors, there must be at least two to combine"
    assert len(labels) >= 2, msg
    L_F = []
    for l, fp in enumerate(labels):
        L_F += fp()
    return L_F


def get_keyvaluepair(c=[], fpv_name='None'):
    """Get a list of the key_value_pairs target names/values."""
    if len(c) == 0:
        return ['kvp_' + fpv_name]
    else:
        out = []
        for atoms in c:
            field_value = float(atoms['key_value_pairs'][fpv_name])
            out.append(field_value)
        return out


def return_fpv(candidates, fpv_names):
    """Sequentially combine fingerprint vectors. Padding handled automatically.

    Parameters
    ----------
    candidates : list or dict
        Atoms objects to construct fingerprints for.
    fpv_name : list of / single fpv class(es)
        List of fingerprinting classes.

    Returns
    -------
    fingerprint_vector : ndarray
      Fingerprint array (n, m) where n is the number of candidates and m is the
      summed number of features from all fingerprint classes supplied.
    """
    if not isinstance(candidates, (list, defaultdict)):
        raise TypeError("return_fpv requires a list or dict of atoms")

    if not isinstance(fpv_names, list):
        fpv_names = [fpv_names]
    fpvn = len(fpv_names)

    # Find the maximum number of atoms and atomic species.
    maxatoms = np.argmax([len(atoms) for atoms in candidates])
    maxcomp = np.argmax(
        [len(set(atoms.get_chemical_symbols())) for atoms in candidates])

    # PATCH: Ideally fp length would be called from a property.
    fps = np.zeros(fpvn, dtype=int)
    for i, fp in enumerate(fpv_names):
        fps[i] = max(len(fp(candidates[maxcomp])),
                     len(fp(candidates[maxatoms])))

    fingerprint_vector = np.zeros((len(candidates), sum(fps)))
    for i, atoms in tqdm(enumerate(candidates)):
        fingerprint_vector[i] = _get_fpv(atoms, fpv_names, fps)

    return fingerprint_vector


def _get_fpv(atoms, fpv_names, fps):
    """Get the fingerprint vector as an array.

    Parameters
    ----------
    atoms : object
        A single atoms object.
    fpv_name : list of / single fpv class(es)
        List of fingerprinting classes.
    fps : list
        List of expected feature vector lengths.

    Returns
    -------
    fingerprint_vector : list
        A feature vector.
    """
    if len(fpv_names) == 1:
        fp = fpv_names[0](atoms=atoms)
        fingerprint_vector = np.zeros((fps[0]))
        fingerprint_vector[:len(fp)] = fp

    else:
        fingerprint_vector = _concatenate_fpv(atoms, fpv_names, fps)

    return fingerprint_vector


def _concatenate_fpv(atoms, fpv_names, fps):
    """Join multiple fingerprint vectors.

    Parameters
    ----------
    atoms : object
        A single atoms object.
    fpv_name : list of / single fpv class(es)
        List of fingerprinting classes.
    fps : list
        List of expected feature vector lengths.

    Returns
    -------
    fingerprint_vector : list
        A feature vector.
    """
    # Define full feature vector.
    fingerprint_vector = np.zeros((sum(fps)))
    start = 0
    # Iterate through the feature generators and update feature vector.
    for i, name in enumerate(fpv_names):
        fp = name(atoms=atoms)
        fingerprint_vector[start:start + len(fp)] = fp
        start = sum(fps[:i + 1])

    return fingerprint_vector
