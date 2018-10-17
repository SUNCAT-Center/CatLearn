"""Base class for the feature generators.

This is inherited by the other fingerprint generators and allows access to a
number of useful and commonly used functions. Standard functionality that is
implemented and applicable to more than one of the other classes should be
put here.
"""
from catlearn.api.ase_atoms_api import extend_atoms_class
from catlearn.utilities.neighborlist import ase_neighborlist


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

    def make_neighborlist(self, candidate, neighbor_number=1):
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
            nl = ase_neighborlist(candidate)
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


def check_labels(labels, result, atoms):
    """Check that two lists have the same length. If not, print an informative
    error message containing a databse id if present.

    Parameters
    ----------
    labels : list
        A list of feature names.
    result : list
        A fingerprint.
    atoms : object
        A single atoms object.
    """
    if len(result) != len(labels):
        msg = str(len(labels)) + '/' + str(len(result)) + \
            ' labels/fingerprint mismatch.'
        if 'id' in atoms.info:
            msg += ' database id: ' + str(atoms.info['id'])
            msg += ' ' + ' '.join([str(label) for label in labels])
            msg += ' ' + ' '.join([str(value) for value in result])
        raise AssertionError(msg)
