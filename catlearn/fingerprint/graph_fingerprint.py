"""Functions to build a neighbor matrix feature representation."""
from __future__ import absolute_import
from __future__ import division

import json
import numpy as np
import warnings

from catlearn import __path__ as catlearn_path
from catlearn.utilities.neighborlist import catlearn_neighborlist
from .base import BaseGenerator


class GraphFingerprintGenerator(BaseGenerator):
    """Function to build a fingerprint vector based on an atoms object."""

    def __init__(self, **kwargs):
        """Standard fingerprint generator setup.

        Parameters
        ----------
        atom_types : list
            Unique atomic types in the systems. Types are denoted by atomic
            number e.g. for CH4 set [1, 6].
        atom_len : int
            The maximum length of all atomic systems that will be passed in a
            data set.
        element_parameters : str, list
            Optional variable to be passed if property_matrix is to be called.
            Type of atomic parameter upon which to compile the feature vector.
            A full list of atomic parameters can be found here:
            https://pypi.python.org/pypi/mendeleev/
        max_neighbors = str, int
            Maximum number of neighbor shells to account for. Can be int or
            'full', for all possible shells.
        """
        if not hasattr(self, 'atom_types'):
            self.atom_types = kwargs.get('atom_types')
        if not hasattr(self, 'atom_len'):
            self.atom_len = kwargs.get('atom_len')
        if not hasattr(self, 'element_parameters'):
            self.element_parameters = kwargs.get('element_parameters')
        if not hasattr(self, 'max_neighbors'):
            self.max_neighbor = kwargs.get('max_neighbors', 1)

        if not hasattr(self, 'element_data'):
            # Load the Mendeleev parameter data into memory
            with open('/'.join(catlearn_path[0].split('/')[:-1]) +
                      '/catlearn/data/proxy-mendeleev.json') as f:
                self.element_data = json.load(f)

        super(GraphFingerprintGenerator, self).__init__(**kwargs)

    def neighbor_sum_vec(self, data):
        """Transform neighborlist into a neighbor sum feature vector.

        Parameters
        ----------
        data : object
            Target data object from which to generate features.

        Returns
        -------
        features : array
            A 1d numpy array of the feature vector.
        """
        features = self._initialize_features(data)

        con = self._normalize_neighbors(data)

        for i, ep in enumerate(self.element_parameters):
            # Define legth of current descriptor set.
            start_index = i * self.atom_len
            end_index = start_index + self.atom_len

            # Generate set of descriptors
            pro = self._prop2matrix(data, ep)
            result = np.dot(con, pro)

            # Assign results to correct indices in feature array.
            features[start_index:end_index] = np.sort(
                np.sum(result, axis=1))[::-1]

        return features

    def neighbor_mean_vec(self, data):
        """Transform neighborlist into a neighbor averaged feature vector.

        Parameters
        ----------
        data : object
            Target data object from which to generate features.

        Returns
        -------
        features : array
            A 1d numpy array of the feature vector.
        """
        # Make checks and initialize empty feature vector.
        features = self._initialize_features(data)

        con = self._normalize_neighbors(data)

        for i, ep in enumerate(self.element_parameters):
            # Define legth of current descriptor set.
            start_index = i * self.atom_len
            end_index = start_index + self.atom_len

            # Generate set of descriptors
            pro = self._prop2matrix(data, ep)
            result = np.dot(con, pro)

            # Assign results to correct indices in feature array.
            features[start_index:end_index] = np.sort(
                np.mean(result, axis=1))[::-1]

        return features

    def _initialize_features(self, data):
        """Function to perform some checks and initialize empty feature vector.

        Parameters
        ----------
        data : object
            Target data object from which to generate features.

        Returns
        -------
        features : array
            Empty 1d numpy array for feature vector.
        """
        msg = 'element_parameters variable must be set.'
        assert self.element_parameters is not None, msg

        # Check parameters are iterable.
        if not isinstance(self.element_parameters, list):
            self.element_parameters = [self.element_parameters]

        # Initialize feature vector.
        features = np.zeros(self.atom_len * len(self.element_parameters))

        return features

    def _normalize_neighbors(self, data):
        """Function to invert importance of neighbor shells.

        The `catlearn_neighborlist` function returns an array with the neighbor
        shell of atom pairs. The further away two atoms are, the larger the
        number of the neighbor shell. This inverts this relationship so atoms
        that are close to oneanother have larger values.

        Parameters
        ----------
        data : object
            Target data object from which to generate features.

        Returns
        -------
        connection_matrix : array
            The shell weighted connection_matrix.
        """
        # There will be some divide by zero warnings that are handled later.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Calculate the matrix representation.
            connection_matrix = catlearn_neighborlist(
                data, max_neighbor=self.max_neighbor)

            # Invert scale of neighbor shells.
            connection_matrix = np.max(connection_matrix) / connection_matrix
            # Replace inf values from zero divide.
            np.place(connection_matrix, connection_matrix == np.inf, 0.)

        con = np.zeros((self.atom_len, self.atom_len))
        con[:len(data), :len(data)] = connection_matrix

        return con

    def _prop2matrix(self, data, prop):
        """Generate a property matrix based on the atomic types.

        Parameters
        ----------
        data : object
            Target data object.
        property : str
            The target property from mendeleev.

        Returns
        -------
        matrix : array
            The adjacency matrix based with atomic properties included.
        """
        ano = self.get_atomic_numbers(data)

        matrix = np.zeros((self.atom_len, self.atom_len), dtype='f')

        atomic_prop = {}
        for a in set(ano):
            atomic_prop[a] = self.element_data[str(a)].get(prop)

        diag = np.diag([atomic_prop[a] for a in ano])
        d, f = np.shape(diag)
        matrix[:d, :f] = diag

        return matrix
