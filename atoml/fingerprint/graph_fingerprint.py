"""Functions to build a neighbor matrix feature representation."""
from __future__ import absolute_import
from __future__ import division

import json
import numpy as np

from atoml import __path__ as atoml_path
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
        """
        if not hasattr(self, 'atom_types'):
            self.atom_types = kwargs.get('atom_types')
        if not hasattr(self, 'atom_len'):
            self.atom_len = kwargs.get('atom_len')
        if not hasattr(self, 'element_parameters'):
            self.element_parameters = kwargs.get('element_parameters')

        if not hasattr(self, 'element_data'):
            # Load the Mendeleev parameter data into memory
            with open('/'.join(atoml_path[0].split('/')[:-1]) +
                      '/atoml/data/proxy-mendeleev.json') as f:
                self.element_data = json.load(f)

        super(GraphFingerprintGenerator, self).__init__(**kwargs)

    def neighbor_sum_vec(self, data):
        """Transform neighborlist into a neighbor sum feature vector.

        Parameters
        ----------
        data : object
            Target data object from which to get the neighborlist dict.
        """
        msg = 'element_parameters variable must be set.'
        assert self.element_parameters is not None, msg

        # Check parameters are iterable.
        if not isinstance(self.element_parameters, list):
            self.element_parameters = [self.element_parameters]

        # Initialize feature vector.
        features = np.zeros(self.atom_len * len(self.element_parameters))

        # Calculate the matrix representation.
        con = self._dict2matrix(data)
        for i, ep in enumerate(self.element_parameters):
            pro = self._prop2matrix(data, ep)
            result = np.dot(con, pro)

            features[i * self.atom_len:(i + 1) * self.atom_len] = np.sort(
                np.sum(result, axis=1))[::-1]

        return features

    def _dict2matrix(self, data):
        """Transform neighborlist dict to binary matrix.

        Parameters
        ----------
        data : object
            Target data object from which to get the neighborlist dict.
        """
        # Generate the setup data first.
        if self.atom_len is None:
            self.atom_len = len(data)

        # Get the neighbor list.
        nl = self.get_neighborlist(data)

        # Initialize the matrix of correct size.
        matrix = np.zeros((self.atom_len, self.atom_len), dtype=np.float32)

        for index in nl:
            for neighbor in nl[index]:
                matrix[index][neighbor] = 1.

        return matrix

    def _prop2matrix(self, data, prop):
        """Generate a property matrix based on the atomic types.

        Parameters
        ----------
        aata : object
            Target data object.
        property : str
            The target property from mendeleev.
        """
        ano = self.get_atomic_numbers(data)

        matrix = np.zeros((self.atom_len, self.atom_len), dtype=np.float32)

        atomic_prop = {}
        for a in set(ano):
            atomic_prop[a] = self.element_data[str(a)].get(prop)

        diag = np.diag([atomic_prop[a] for a in ano])
        d, f = np.shape(diag)
        matrix[:d, :f] = diag

        return matrix
