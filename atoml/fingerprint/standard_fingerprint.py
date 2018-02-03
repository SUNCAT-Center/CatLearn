"""Standard fingerprint functions."""
from __future__ import absolute_import
from __future__ import division

import numpy as np

from .base import FeatureGenerator


class StandardFingerprintGenerator(FeatureGenerator):
    """Function to build a fingerprint vector based on an atoms object."""

    def __init__(self, atom_types=None, dtype='atoms'):
        """Standard fingerprint generator setup.

        Parameters
        ----------
        atom_types : list
            Unique atomic types in the systems.
        """
        self.atom_types = atom_types
        FeatureGenerator.__init__(self, dtype=dtype)

    def mass_fpv(self, candidate):
        """Function that takes a list of atoms objects and returns the mass."""
        # Return the summed mass of the atoms object.
        return np.array([sum(self.get_masses(candidate))])

    def composition_fpv(self, candidate):
        """Basic function to take atoms object and return the composition."""
        an = self.get_atomic_numbers(candidate)
        # Generate a list of atom types if not supplied.
        if self.atom_types is None:
            self.atom_types = sorted(frozenset(an))

        # Add count of each atom type to the fingerprint vector.
        return np.array([an.count(i) for i in self.atom_types])

    def _get_coulomb(self, candidate):
        """Generate the coulomb matrix.

        A more detailed discussion of the coulomb features can be found here:
        https://doi.org/10.1103/PhysRevLett.108.058301

        Parameters
        ----------
        candidate : object
          Data object with Cartesian coordinates and atomic numbers available.

        Returns
        -------
        coulomb : ndarray
          The coulomb matrix, (n, n) atoms in size.
        """
        if len(candidate) < 2:
            raise ValueError(
                ("Columb matrix requires atoms object with at least 2 atoms"))

        dm = self.get_all_distances(candidate)
        np.fill_diagonal(dm, 1.)

        # Make coulomb matrix
        an = self.get_atomic_numbers(candidate)
        coulomb = np.outer(an, an) / dm

        # Set diagonal elements
        r = range(len(an))
        coulomb[r, r] = 0.5 * an ** 2.4

        return coulomb

    def eigenspectrum_fpv(self, candidate):
        """Sorted eigenspectrum of the Coulomb matrix.

        Parameters
        ----------
        candidate : object
          Data object with Cartesian coordinates and atomic numbers available.

        Returns
        -------
        result : ndarray
          Sorted Eigen values of the coulomb matrix, n atoms is size.
        """
        coulomb = self._get_coulomb(candidate)

        w, _ = np.linalg.eig(coulomb)

        return np.sort(w)[::-1]

    def distance_fpv(self, candidate):
        """Averaged distance between e.g. A-A atomic pairs."""
        fp = []
        an = self.get_atomic_numbers(candidate)
        pos = self.get_positions(candidate)
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
