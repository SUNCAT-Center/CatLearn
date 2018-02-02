"""Functions that interface ase with AtoML."""
import types


def extend_class(atoms):
    """A wrapper to add extra functionality to ase atoms objects.

    Parameters
    ----------
    atoms : class
        An ase atoms object.
    """
    atoms.store_fp_in_atoms = types.MethodType(store_fp_in_atoms, atoms)
    atoms.load_fp_in_atoms = types.MethodType(load_fp_in_atoms, atoms)


def set_features(self, fp):
    """Function to write feature vector to ase atoms object.

    This function provides a uniform way in which to attach a feature vector to
    an atoms object. Can be used in conjunction with the `get_features`
    function.

    Parameters
    ----------
    self : class
        An ase atoms object to attach feature vector to.
    fp : array
        The feature vector to attach.
    """
    if 'atoml' not in self.info:
        self.info['atoml'] = {}
    self.info['atoml']['features'] = fp


def get_features(self):
    """Function to read feature vector from ase atoms object.

    This function provides a uniform way in which to attach a feature vector to
    an atoms object. Can be used in conjunction with the `set_features`
    function.

    Parameters
    ----------
    self : class
        An ase atoms object to attach feature vector to.

    Returns
    -------
    fp : array
        The feature vector attach to the atoms object.
    """
    return self.info['atoml']['features']
