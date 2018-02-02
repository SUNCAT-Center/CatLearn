"""Functions that interface ase with AtoML."""
import types


def extend_class(atoms):
    """A wrapper to add extra functionality to ase atoms objects.

    Parameters
    ----------
    atoms : class
        An ase atoms object.
    """
    atoms._initialize_atoml = types.MethodType(_initialize_atoml, atoms)
    atoms.set_features = types.MethodType(set_features, atoms)
    atoms.get_features = types.MethodType(get_features, atoms)


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
    self._initialize_atoml()
    self.atoml['features'] = fp


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
    self._initialize_atoml()
    try:
        return self.atoml['features']
    except KeyError:
        return None


def _initialize_atoml(self):
    """A base function to initialize the atoml functionality."""
    try:
        self.atoml
    except AttributeError:
        self.atoml = {}
