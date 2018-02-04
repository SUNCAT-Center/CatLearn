"""Base class for the feature generators."""
import numpy as np
from collections import defaultdict

from ase.data import covalent_radii

from atoml.api.ase_api import extend_atoms_class


class BaseGenerator(object):
    """Base class for feature generation."""

    def __init__(self, **kwargs):
        """Initialize the class.

        Parameters
        ----------
        dtype : str
            Expected data type. Currently only supports ase atoms objects.
        """
        self.dtype = kwargs.get('dtype', 'atoms')

    def make_neighborlist(self, candidate, dx=None, neighbor_number=1):
        """Function to generate the neighborlist.

        Parameters
        ----------
        candidate : object
            Target data object on which to generate neighbor list.
        dx : dict
            Buffer to calculate nearest neighbor pairs in dict format:
            dx = {atomic_number: buffer}.
        neighbor_number : int
            Neighbor shell.
        """
        if self.dtype == 'atoms':
            extend_atoms_class(candidate)
            nl = self.atoms_neighborlist(candidate, dx, neighbor_number)
        else:
            raise NotImplementedError('{} data type not implemented.'.format(
                self.dtype))

        candidate.set_neighborlist(nl)

        return nl

    def get_neighborlist(self, candidate):
        """Function to return the neighborlist.

        It will check to see if the neighbor list is stored in the data object.
        If not it will generate the neighborlist from scratch.

        Parameters
        ----------
        candidate : object
            Target data object from which to get the neighbor list.
        """
        try:
            nl = candidate.get_neighborlist()
        except AttributeError:
            nl = None

        if nl is None:
            nl = self.make_neighborlist(candidate)

        return nl

    def get_positions(self, candidate):
        """Function to return the atomic coordinates.

        Parameters
        ----------
        candidate : object
            Target data object from which to get the atomic coordinates.
        """
        if self.dtype == 'atoms':
            return candidate.get_positions()
        else:
            raise NotImplementedError('{} data type not implemented.'.format(
                self.dtype))

    def get_atomic_numbers(self, candidate):
        """Function to return the atomic numbers.

        Parameters
        ----------
        candidate : object
            Target data object from which to get the atomic numbers.
        """
        if self.dtype == 'atoms':
            return candidate.get_atomic_numbers()
        else:
            raise NotImplementedError('{} data type not implemented.'.format(
                self.dtype))

    def get_masses(self, candidate):
        """Function to return the atomic masses.

        Parameters
        ----------
        candidate : object
            Target data object from which to get the atomic masses.
        """
        if self.dtype == 'atoms':
            return candidate.get_masses()
        else:
            raise NotImplementedError('{} data type not implemented.'.format(
                self.dtype))

    def get_all_distances(self, candidate):
        """Function to return the atomic distances.

        Parameters
        ----------
        candidate : object
            Target data object from which to get the atomic distances.
        """
        if self.dtype == 'atoms':
            return candidate.get_all_distances()
        else:
            raise NotImplementedError('{} data type not implemented.'.format(
                self.dtype))

    def atoms_neighborlist(self, atoms, dx=None, neighbor_number=1):
        """Make dict of neighboring atoms for discrete system.

        Possible to return neighbors from defined neighbor shell e.g. 1st, 2nd,
        3rd by changing the neighbor number.

        Parameters
        ----------
        atoms : object
            Target ase atoms object on which to get neighbor list.
        dx : dict
            Buffer to calculate nearest neighbor pairs in dict format:
            dx = {atomic_number: buffer}.
        neighbor_number : int
            Neighbor shell.
        """
        # Set up buffer dict.
        if dx is None:
            dx = dict.fromkeys(set(atoms.get_atomic_numbers()), 0)
            for i in dx:
                dx[i] = covalent_radii[i] / 2.

        conn = defaultdict(list)
        for a1 in atoms:
            for a2 in atoms:
                if a1.index != a2.index:
                    d = np.linalg.norm(np.asarray(a1.position) -
                                       np.asarray(a2.position))
                    r1 = covalent_radii[a1.number]
                    r2 = covalent_radii[a2.number]
                    dxi = (dx[a1.number] + dx[a2.number]) / 2.
                    if neighbor_number == 1:
                        d_max1 = 0.
                    else:
                        d_max1 = ((neighbor_number - 1) * (r2 + r1)) + dxi
                    d_max2 = (neighbor_number * (r2 + r1)) + dxi
                    if d > d_max1 and d < d_max2:
                        conn[a1.index].append(a2.index)
        return conn
