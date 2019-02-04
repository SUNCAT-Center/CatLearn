"""Functions to setup fingerprint vectors."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd

from collections import defaultdict
import multiprocessing
from tqdm import tqdm
from catlearn.fingerprint.adsorbate import (AdsorbateFingerprintGenerator,
                                            default_adsorbate_fingerprinters)
from catlearn.fingerprint.convoluted import (ConvolutedFingerprintGenerator,
                                             default_convoluted_fingerprinters)
from catlearn.fingerprint.particle import (ParticleFingerprintGenerator,
                                           default_particle_fingerprinters)
from catlearn.fingerprint.standard import (StandardFingerprintGenerator,
                                           default_molecule_fingerprinters)
from catlearn.fingerprint.graph import GraphFingerprintGenerator
from catlearn.fingerprint.bulk import (BulkFingerprintGenerator,
                                       default_bulk_fingerprinters)
from catlearn.fingerprint.chalcogenide import \
    ChalcogenideFingerprintGenerator, default_chalcogenide_fingerprinters
from catlearn.fingerprint.catapp import CatappFingerprintGenerator

default_sets = {'bulk': default_bulk_fingerprinters,
                'fragment': (default_molecule_fingerprinters +
                             default_particle_fingerprinters),
                'adsorbates': (default_adsorbate_fingerprinters +
                               default_convoluted_fingerprinters),
                'chalcogenides': default_chalcogenide_fingerprinters}


def default_fingerprinters(generator, data_type):
    """"Return a list of generators.

    Parameters
    ----------
    generator : object
        FeatureGenerator object
    data_type : str
        'bulk', 'adsorbates' or 'fragment'

    Returns
    ----------
    vec_name : list of / single vec class(es)
        List of fingerprinting classes.
    """
    allowed = list(default_sets.keys())
    if data_type not in allowed:
        msg = "data type must be " + " or ".join(allowed)
        msg += ". 'fragment' is for anything without periodic boundary cond."
        raise ValueError(msg)
    return [getattr(generator, name) for name in default_sets[data_type]]


class FeatureGenerator(
        AdsorbateFingerprintGenerator, ParticleFingerprintGenerator,
        StandardFingerprintGenerator, GraphFingerprintGenerator,
        BulkFingerprintGenerator, ConvolutedFingerprintGenerator,
        ChalcogenideFingerprintGenerator, CatappFingerprintGenerator):
    """Feature generator class.

    It is sometimes necessary to normalize the length of feature vectors when
    data is supplied with variable numbers of atoms or elemental types. If this
    is the case, use the `normalize_features` function.

    In this class, there are functions to take a data object and return a
    feature vector. This is done with the `return_vec` function. The names of
    the descriptors in the feature vector can be accessed with the
    `return_names` function.

    The class inherits the actual generator functions from the
    [NAME]FingerprintGenerator classes. Additional variables are passed as
    kwargs.
    """

    def __init__(self, atom_types=None, atom_len=None, nprocs=1, **kwargs):
        """Initialize feature generator.

        Parameters
        ----------
        atom_types : list
            Unique atomic types in the systems. Types are denoted by atomic
            number e.g. for CH4 set [1, 6].
        atom_len : int
            The maximum length of all atomic systems that will be passed in a
            data set.
        nprocs : int
            Number of cores available for parallelization. Default is 1, e.g.
            serial. Set None to use all available cores.
        """
        self.atom_types = atom_types
        self.atom_len = atom_len
        self.nprocs = nprocs

        super(FeatureGenerator, self).__init__(**kwargs)

    def normalize_features(self, train_candidates, test_candidates=None):
        """Function to attach feature data to class.

        Currently the function attaches data on all elemental types present in
        the data as well as the maximum number of atoms in a data object.

        Parameters
        ----------
        train_candidates : list
            List of atoms objects.
        test_candidates : list
            List of atoms objects.
        """
        self._get_atom_types(train_candidates, test_candidates)
        self._get_atom_length(train_candidates, test_candidates)

    def featurize_atomic_pairs(self, candidates):
        """Featurize pairs of atoms by their elements and pair distances, in
        order to optimize the bond classifier.

        Parameters
        ----------
        candidates : list of atoms objects.

        Returns
        ----------
        data : array
            Data matrix.
        """
        # Find the atomic species in data if needed.
        if self.atom_types is None:
            self._get_atom_types(candidates)

        data = []
        for atoms in tqdm(candidates):
            # One hot encode elements.
            dummies = pd.get_dummies(atoms.numbers)
            d = dummies.T.reindex(self.atom_types).fillna(0).T.as_matrix()

            # Number of pairs by 1 + number of element types data matrix.
            n_pairs = len(atoms) ** 2
            fingerprint = np.empty([n_pairs, len(self.atom_types) + 3])
            # Pair distances in first column.
            fingerprint[:, 0] = atoms.pair_distances.ravel()
            # Occurrence of element types in the following columns.
            for i, pair in enumerate(fingerprint):
                row, col = np.unravel_index(i, [len(atoms), len(atoms)])
                fingerprint[i, -2:] = np.array([atoms.numbers[row],
                                                atoms.numbers[col]])
                fingerprint[i, 1:-2] = d[row, :] + d[col, :]

            data.append(fingerprint.tolist())

        return np.concatenate(data, axis=0)

    def return_vec(self, candidates, vec_names):
        """Sequentially combine feature vectors. Padding handled automatically.

        Parameters
        ----------
        candidates : list or dict
            Atoms objects to construct fingerprints for.
        vec_name : list
            List of fingerprinting functions.

        Returns
        -------
        vector : ndarray
          Fingerprint array (n, m) where n is the number of candidates and m is
          the summed number of features from all fingerprint classes supplied.
        """
        # Check for a list or dict of atomic data.
        if not isinstance(candidates, (list, defaultdict)):
            raise TypeError("return_vec requires a list or dict of atoms")

        if not isinstance(vec_names, list):
            vec_names = [vec_names]

        # Find the maximum number of atomic species in data if needed.
        if self.atom_types is None:
            self._get_atom_types(candidates)
        # Find the maximum number of atoms in data if needed.
        if self.atom_len is None:
            self._get_atom_length(candidates)

        fingerprint_vector = []
        args = tuple((atoms, vec_names) for atoms in candidates)

        # Check for parallelized feature generation.
        if self.nprocs != 1:
            pool = multiprocessing.Pool(self.nprocs)
            parallel_iterate = pool.map_async(
                self._get_vec, args, callback=fingerprint_vector.append)
            parallel_iterate.wait()
            vector = np.asarray(fingerprint_vector)[0]
        else:
            for a in tqdm(args):
                fingerprint_vector.append(self._get_vec(a))
            vector = np.asarray(fingerprint_vector)

        return vector

    def return_names(self, vec_names):
        """Function to return a list of feature names.

        Parameters
        ----------
        vec_name : list
            List of fingerprinting functions.

        Returns
        -------
        fingerprint_vector : ndarray
            Name array.
        """
        if not isinstance(vec_names, list):
            vec_names = [vec_names]

        if len(vec_names) == 1:
            return vec_names[0](None)
        else:
            return self._concatenate_vec(None, vec_names)

    def _get_vec(self, args):
        """Get the fingerprint vector as an array.

        Parameters
        ----------
        atoms : object
            A single atoms object.
        vec_name : list of / single vec class(es)
            List of fingerprinting classes.
        fps : list
            List of expected feature vector lengths.

        Returns
        -------
        fingerprint_vector : list
            A feature vector.
        """
        atoms, vec_names = args
        if len(vec_names) == 1:
            return vec_names[0](atoms)
        else:
            return self._concatenate_vec(atoms, vec_names)

    def _concatenate_vec(self, atoms, vec_names):
        """Join multiple fingerprint vectors.

        Parameters
        ----------
        atoms : object
            A single atoms object.
        vec_name : list
            List of fingerprinting functions.
        fps : list
            List of expected feature vector lengths.

        Returns
        -------
        fingerprint_vector : list
            A feature vector.
        """
        fingerprint_vector = np.array([])
        # Iterate through the feature generators and update feature vector.
        for name in vec_names:
            arr = np.array(name(atoms))
            # Do not allow concatenation of str types with other types.
            if len(fingerprint_vector) > 0:
                if ((arr.dtype.type is np.str_ and
                     fingerprint_vector.dtype.type is not np.str_) or
                    (arr.dtype.type is not np.str_ and
                     fingerprint_vector.dtype.type is np.str_)):
                        raise AssertionError("Fingerprints should be float" +
                                             " type, and returned separately" +
                                             " from str types.")
            fingerprint_vector = np.concatenate((fingerprint_vector,
                                                 arr))

        return fingerprint_vector

    def _get_atom_types(self, train_candidates, test_candidates=None):
        """Function to get all potential atomic types in data.

        Parameters
        ----------
        train_candidates : list
            List of atoms objects.
        test_candidates : list
            List of atoms objects.

        Returns
        -------
        atom_types : list
            Full list of atomic numbers in data.
        """
        train_candidates = list(train_candidates)
        if test_candidates is not None:
            train_candidates += list(test_candidates)
        atom_types = set()
        for a in train_candidates:
            atom_types.update(set(a.get_atomic_numbers()))
        atom_types = sorted(list(atom_types))

        self.atom_types = atom_types

    def _get_atom_length(self, train_candidates, test_candidates=None):
        """Function to get all potential system sizes in data.

        Parameters
        ----------
        train_candidates : list
            List of atoms objects.
        test_candidates : list
            List of atoms objects.

        Returns
        -------
        atom_types : list
            Full list of system sizes in data.
        """
        train_candidates = list(train_candidates)
        if test_candidates is not None:
            train_candidates += list(test_candidates)

        max_len = 0
        for a in train_candidates:
            if max_len < len(a):
                max_len = len(a)

        self.atom_len = max_len

    def get_dataframe(self, candidates, vec_names):
        """Sequentially combine feature vectors. Padding handled automatically.

        Parameters
        ----------
        candidates : list or dict
            Atoms objects to construct fingerprints for.
        vec_name : list
            List of fingerprinting functions.

        Returns
        -------
        df : DataFrame
          Fingerprint dataframe with n rows and m columns (n, m) where
          n is the number of candidates and m is the summed number of features
          from all fingerprint classes supplied.
        """
        matrix = self.return_vec(candidates, vec_names)
        columns = self.return_names(vec_names)

        return pd.DataFrame(data=matrix, columns=columns)
