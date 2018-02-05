"""Standard fingerprint functions."""
from __future__ import absolute_import
from __future__ import division

import json
import numpy as np

from atoml import __path__ as atoml_path
from .base import BaseGenerator


class StandardFingerprintGenerator(BaseGenerator):
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
            Optional variable to be passed if element_parameter_vec is to be
            called. Type of atomic parameter upon which to compile the feature
            vector. A full list of atomic parameters can be found here:
            https://pypi.python.org/pypi/mendeleev/
        """
        if not hasattr(self, 'atom_types'):
            self.atom_types = kwargs.get('atom_types')
        if not hasattr(self, 'atom_len'):
            self.atom_len = kwargs.get('atom_len')
        self.element_parameters = kwargs.get('element_parameters')

        # Load the Mendeleev parameter data into memory
        with open('/'.join(atoml_path[0].split('/')[:-1]) +
                  '/atoml/data/proxy-mendeleev.json') as f:
            self.element_data = json.load(f)

        super(StandardFingerprintGenerator, self).__init__(**kwargs)

    def composition_vec(self, data):
        """Function to return a feature vector based on the composition.

        Parameters
        ----------
        data : object
            Data object with atomic numbers available.

        Returns
        -------
        features : array
            Vector containing a count of the different atomic types, e.g. for
            CH3OH the vector [1, 4, 1] would be returned.
        """
        # Return feature names in no atomic data is passed.
        if data is None:
            msg = 'Class must have atom_types set to return feature names.'
            assert hasattr(self, 'atom_types') and self.atom_types is not \
                None, msg
            return ['{}_count'.format(n) for n in self.atom_types]

        ano = self.get_atomic_numbers(data)

        # WARNING: Will be set permanently whichever atom is first passed.
        if self.atom_types is None:
            self.atom_types = sorted(frozenset(ano))

        return np.array([list(ano).count(sym) for sym in self.atom_types])

    def element_parameter_vec(self, data):
        """Function to return a vector based on a defined paramter.

        The vector is compiled based on the summed parameters for each
        elemental type as well as the sum for all atoms.

        Parameters
        ----------
        data : object
            Data object with atomic numbers available.

        Returns
        -------
        features : array
            An n + 1 array where n in the length of self.atom_types.
        """
        msg = 'The variable element_parameters must be set in the feature '
        msg += 'generator class.'
        assert self.element_parameters is not None, msg

        # Return feature names in no atomic data is passed.
        if data is None:
            msg = 'Class must have atom_types set to return feature names.'
            assert hasattr(self, 'atom_types') and self.atom_types is not \
                None, msg
            names = []
            for p in self.element_parameters:
                names += ['sum_{0}_{1}'.format(n, p) for n in self.atom_types]
                names += ['sum_all_{}'.format(p)]
            return names

        if not isinstance(self.element_parameters, list):
            self.element_parameters = [self.element_parameters]

        # Get the composition data.
        comp = self.composition_vec(data)

        # Generate the actual features.
        features = np.asarray([])
        for p in self.element_parameters:
            plist = [self.element_data[str(an)].get(p) for an in
                     self.atom_types]

            f = np.zeros(len(comp) + 1)
            f[:len(comp)] = np.multiply(comp, plist)
            f[-1] = np.sum(features)
            features = np.concatenate((features, f))

        return features

    def element_mass_vec(self, data):
        """Function to return a vector based on mass parameter."""
        # Return feature names in no atomic data is passed.
        if data is None:
            return ['sum_mass']
        # Return the summed mass of the atoms object.
        return np.array([sum(self.get_masses(data))])

    def _get_coulomb(self, data):
        """Generate the coulomb matrix.

        A more detailed discussion of the coulomb features can be found here:
        https://doi.org/10.1103/PhysRevLett.108.058301

        Parameters
        ----------
        data : object
            Data object with Cartesian coordinates and atomic numbers
            available.

        Returns
        -------
        coulomb : ndarray
            The coulomb matrix, (n, n) atoms in size.
        """
        if len(data) < 2:
            raise ValueError(
                'Columb matrix requires atoms object with at least 2 atoms')

        dm = self.get_all_distances(data)
        np.fill_diagonal(dm, 1)

        # Make coulomb matrix
        ano = self.get_atomic_numbers(data)
        coulomb = np.outer(ano, ano) / dm

        diagonal = 0.5 * ano ** 2.4
        np.fill_diagonal(coulomb, diagonal)

        return coulomb

    def eigenspectrum_vec(self, data):
        """Sorted eigenspectrum of the Coulomb matrix.

        Parameters
        ----------
        data : object
          Data object with Cartesian coordinates and atomic numbers available.

        Returns
        -------
        features : ndarray
          Sorted Eigen values of the coulomb matrix, n atoms is size.
        """
        # Return feature names in no atomic data is passed.
        if data is None:
            msg = 'Class must have atom_len set to return feature names.'
            assert hasattr(self, 'atom_len') and self.atom_len is not \
                None, msg
            return ['eig_{}'.format(n) for n in range(self.atom_len)]

        features = np.zeros(self.atom_len)
        coulomb = self._get_coulomb(data)

        v = np.linalg.eigvals(coulomb)
        v[::-1].sort()
        features[:len(v)] = v

        return features

    def distance_vec(self, data):
        """Averaged distance between e.g. A-A atomic pairs."""
        # Return feature names in no atomic data is passed.
        if data is None:
            msg = 'Class must have atom_types set to return feature names.'
            assert hasattr(self, 'atom_types') and self.atom_types is not \
                None, msg
            return ['{0}-{0}_dist'.format(n) for n in self.atom_types]

        fp = []
        an = self.get_atomic_numbers(data)
        pos = self.get_positions(data)
        if self.atom_types is None:
            # Get unique atom types.
            self.atom_types = sorted(frozenset(an))
        for at in self.atom_types:
            ad = 0.
            co = 0
            for i, j in zip(an, pos):
                if i == at:
                    for k, l in zip(an, pos):
                        if k == at and all(j != l):
                            co += 1
                            ad += np.linalg.norm(j - l)
            if co != 0:
                fp.append(ad / co)
            else:
                fp.append(0.)
        return fp
