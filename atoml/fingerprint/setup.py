"""Functions to setup fingerprint vectors."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from collections import defaultdict

from .particle_fingerprint import ParticleFingerprintGenerator
from .standard_fingerprint import StandardFingerprintGenerator


class FeatureGenerator(
        ParticleFingerprintGenerator, StandardFingerprintGenerator):
    """Feature generator class."""

    def __init__(self, atom_types=None, atom_len=None, dtype='atoms'):
        """Initialize feature generator.

        Parameters
        ----------
        atom_types : list
            Unique atomic types in the systems. Types are denoted by atomic
            number e.g. for CH4 set [1, 6].
        """
        self.atom_types = atom_types
        self.atom_len = atom_len
        self.dtype = dtype

        super(FeatureGenerator, self).__init__()

    def get_combined_descriptors(self, fpv_list):
        """Sequentially combine feature label vectors.

        Parameters
        ----------
        fpv_list : list
            Functions that return fingerprints.
        """
        # Check that there are at least two fingerprint descriptors to combine.
        msg = "This functions combines various fingerprint"
        msg += " vectors, there must be at least two to combine"
        assert len(fpv_list) >= 2, msg
        labels = fpv_list[::-1]
        L_F = []
        for j in range(len(labels)):
            L_F.append(labels[j]())
        return np.hstack(L_F)

    def get_keyvaluepair(self, c=[], fpv_name='None'):
        """Get a list of the key_value_pairs target names/values."""
        if len(c) == 0:
            return ['kvp_' + fpv_name]
        else:
            out = []
            for atoms in c:
                field_value = float(atoms['key_value_pairs'][fpv_name])
                out.append(field_value)
            return out

    def return_fpv(self, candidates, fpv_names):
        """Sequentially combine feature vectors. Padding handled automatically.

        Parameters
        ----------
        candidates : list or dict
            Atoms objects to construct fingerprints for.
        fpv_name : list of / single fpv class(es)
            List of fingerprinting classes.

        Returns
        -------
        fingerprint_vector : ndarray
          Fingerprint array (n, m) where n is the number of candidates and m is
          the summed number of features from all fingerprint classes supplied.
        """
        if not isinstance(candidates, (list, defaultdict)):
            raise TypeError("return_fpv requires a list or dict of atoms")

        if not isinstance(fpv_names, list):
            fpv_names = [fpv_names]

        # Find the maximum number of atomic species in data if needed.
        if self.atom_types is None:
            self._get_atom_types(candidates)
        # Find the maximum number of atoms in data if needed.
        if self.atom_len is None:
            self._get_atom_length(candidates)

        fingerprint_vector = []
        for atoms in candidates:
            fingerprint_vector.append(self._get_fpv(atoms, fpv_names))

        return np.asarray(fingerprint_vector)

    def _get_fpv(self, atoms, fpv_names):
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
            return fpv_names[0](atoms)
        else:
            return self._concatenate_fpv(atoms, fpv_names)

    def _concatenate_fpv(self, atoms, fpv_names):
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
        fingerprint_vector = np.array([])
        # Iterate through the feature generators and update feature vector.
        for name in fpv_names:
            fingerprint_vector = np.concatenate((fingerprint_vector,
                                                 name(atoms)))

        return fingerprint_vector

    def normalize_features(self, train_candidates, test_candidates=None):
        """Function to attach feature data to class.

        Parameters
        ----------
        train_candidates : list
            List of atoms objects.
        test_candidates : list
            List of atoms objects.
        """
        self._get_atom_types(train_candidates, test_candidates)
        self._get_atom_length(train_candidates, test_candidates)

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

        max_len = 0
        for a in train_candidates:
            if max_len < len(a):
                max_len = len(a)

        self.atom_len = max_len
