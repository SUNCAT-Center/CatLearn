"""Standard fingerprint functions."""
from __future__ import absolute_import
from __future__ import division

import numpy as np

no_asap = False
try:
    from asap3.analysis import PTM
except ImportError:
    no_asap = True


class StandardFingerprintGenerator(object):
    """Function to build a fingerprint vector based on an atoms object."""

    def __init__(self, atom_types=None):
        """Standard fingerprint generator setup.

        Parameters
        ----------
        atom_types : list
            Unique atomic types in the systems.
        """
        self.atom_types = atom_types

    def mass_fpv(self, atoms):
        """Function that takes a list of atoms objects and returns the mass."""
        # Return the summed mass of the atoms object.
        return np.array([sum(atoms.get_masses())])

    def composition_fpv(self, atoms):
        """Basic function to take atoms object and return the composition."""
        cs = atoms.get_chemical_symbols()
        # Generate a list of atom types if not supplied.
        if self.atom_types is None:
            self.atom_types = sorted(frozenset(cs))

        # Add count of each atom type to the fingerprint vector.
        return np.array([cs.count(i) for i in self.atom_types])

    def _get_coulomb(self, atoms):
        """Generate the coulomb matrix.

        A more detailed discussion of the coulomb features can be found here:
        https://doi.org/10.1103/PhysRevLett.108.058301

        Parameters
        ----------
        atoms : object
          Atoms object with Cartesian coordinates available.

        Returns
        -------
        coulomb : ndarray
          The coulomb matrix, (n, n) atoms in size.
        """
        if len(atoms) < 2:
            raise ValueError(
                ("Columb matrix requires atoms object with at least 2 atoms"))

        dm = atoms.get_all_distances()
        np.fill_diagonal(dm, 1.)

        # Make coulomb matrix
        coulomb = np.outer(atoms.numbers, atoms.numbers) / dm

        # Set diagonal elements
        r = range(len(atoms))
        coulomb[r, r] = 0.5 * atoms.numbers ** 2.4

        return coulomb

    def eigenspectrum_fpv(self, atoms):
        """Sorted eigenspectrum of the Coulomb matrix.

        Parameters
        ----------
        atoms : object
          Atoms object with Cartesian coordinates available.

        Returns
        -------
        result : ndarray
          Sorted Eigen values of the coulomb matrix, n atoms is size.
        """
        coulomb = self._get_coulomb(atoms)

        w, _ = np.linalg.eig(coulomb)

        return np.sort(w)[::-1]

    def distance_fpv(self, atoms):
        """Averaged distance between e.g. A-A atomic pairs."""
        fp = []
        an = atoms.get_atomic_numbers()
        pos = atoms.get_positions()
        if self.atom_types is None:
            # Get unique atom types.
            self.atom_types = frozenset(an)
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

    def ptm_structure_fpv(self, atoms):
        """Polyhedral Template Matching wrapper for ASAP."""
        msg = "ASAP must be installed to use this function:"
        msg += " https://wiki.fysik.dtu.dk/asap"
        assert not no_asap, msg
        ptmdata = PTM(atoms)
        return ptmdata['structure']

    def ptm_alloy_fpv(self, atoms):
        """Polyhedral Template Matching wrapper for ASAP."""
        msg = "ASAP must be installed to use this function:"
        msg += " https://wiki.fysik.dtu.dk/asap"
        assert not no_asap, msg
        ptmdata = PTM(atoms)
        return ptmdata['alloytype']
