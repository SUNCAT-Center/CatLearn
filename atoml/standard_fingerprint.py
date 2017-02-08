""" Standard fingerprint functions. """
import numpy as np

no_asap = False
try:
    from asap3.analysis import PTM
except ImportError:
    no_asap = True


class StandardFingerprintGenerator(object):
    def __init__(self, atom_types=None):
        """ atom_types: list
                List of all unique atomic types in the systems under
                consideration. Should always be defined if elemental makeup
                varies between candidates to preserve a constant ordering.
        """
        self.atom_types = atom_types

    def mass_fpv(self, atoms):
        """ Function that takes a list of atoms objects and returns the mass.
        """
        # Return the summed mass of the atoms object.
        return np.array([sum(atoms.get_masses())])

    def composition_fpv(self, atoms):
        """ Basic function to take atoms object and return the composition. """
        # Generate a list of atom types if not supplied.
        if self.atom_types is None:
            self.atom_types = frozenset(atoms.get_chemical_symbols())

        # Add count of each atom type to the fingerprint vector.
        fp = []
        for i in self.atom_types:
            count = 0.
            for j in atoms.get_chemical_symbols():
                if i == j:
                    count += 1.
            fp.append(count)
        return np.array(fp)

    def get_coulomb(self, atoms):
        """ Function to generate the coulomb matrix.

        Returns a numpy array.
        """
        # Get distances
        dm = atoms.get_all_distances()
        np.fill_diagonal(dm, 1.)

        # Make coulomb matrix
        coulomb = np.outer(atoms.numbers, atoms.numbers) / dm

        # Set diagonal elements
        r = range(len(atoms))
        coulomb[r, r] = 0.5 * atoms.numbers ** 2.4

        return coulomb

    def eigenspectrum_fpv(self, atoms):
        """ Function that takes a list of atoms objects and returns a list of
            fingerprint vectors in the form of the sorted eigenspectrum of the
            Coulomb matrix as defined in J. Chem. Theory Comput. 2013, 9,
            3404-3419.
        """
        # Get the Coulomb matrix.
        coulomb = self.get_coulomb(atoms)
        # Get eigenvalues and vectors
        w, v = np.linalg.eig((np.array(coulomb)))
        # Return sort eigenvalues from largest to smallest
        return np.sort(w)[::-1]

    def distance_fpv(self, atoms):
        """ Function to calculate the averaged distance between e.g. A-A atomic
            pairs. The distance measure can be useful to describe how close
            atoms preferentially sit in the system.
        """
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
                            ad += np.linalg.norm(j-l)
            if co != 0:
                fp.append(ad / co)
            else:
                fp.append(0.)
        return fp

    def ptm_structure_fpv(self, atoms):
        """ Function that uses the Polyhedral Template Matching routine in ASAP
            to assess the structure and return a list with the crystal
            structure environment of each atom. Greater detail can be found at
            the following:
            https://wiki.fysik.dtu.dk/asap/Local%20crystalline%20order
        """
        msg = "ASAP must be installed to use this function:"
        msg += " https://wiki.fysik.dtu.dk/asap"
        assert not no_asap, msg
        ptmdata = PTM(atoms)
        return ptmdata['structure']

    def ptm_alloy_fpv(self, atoms):
        """ Function that uses the Polyhedral Template Matching routine in ASAP
            to assess the structure and return a list with the alloy structure
            environment of each atom.
        """
        msg = "ASAP must be installed to use this function:"
        msg += " https://wiki.fysik.dtu.dk/asap"
        assert not no_asap, msg
        ptmdata = PTM(atoms)
        return ptmdata['alloytype']
