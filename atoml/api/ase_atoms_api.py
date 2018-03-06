"""Functions that interface ase with AtoML."""
import types


def extend_atoms_class(atoms):
    """A wrapper to add extra functionality to ase atoms objects.

    Parameters
    ----------
    atoms : class
        An ase atoms object.
    """
    atoms._initialize_atoml = types.MethodType(_initialize_atoml, atoms)
    atoms.set_features = types.MethodType(set_features, atoms)
    atoms.get_features = types.MethodType(get_features, atoms)
    atoms.set_neighborlist = types.MethodType(set_neighborlist, atoms)
    atoms.get_neighborlist = types.MethodType(get_neighborlist, atoms)
    atoms.set_graph = types.MethodType(set_graph, atoms)
    atoms.get_graph = types.MethodType(get_graph, atoms)


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

    This function provides a uniform way in which to return a feature vector
    from an atoms object.

    Parameters
    ----------
    self : class
        An ase atoms object to attach feature vector to.

    Returns
    -------
    fp : array
        The feature vector attached to the atoms object.
    """
    self._initialize_atoml()
    try:
        return self.atoml['features']
    except KeyError:
        return None


def set_neighborlist(self, neighborlist):
    """Function to write neighborlist to ase atoms object.

    This function provides a uniform way in which to attach a neighbor list to
    an atoms object. Can be used in conjunction with the `get_neighborlist`
    function.

    Parameters
    ----------
    self : class
        An ase atoms object to attach feature vector to.
    neighborlist : dict
        The neighbor list dict to attach.
    """
    self._initialize_atoml()
    self.atoml['neighborlist'] = neighborlist


def get_neighborlist(self):
    """Function to read neighborlist from ase atoms object.

    This function provides a uniform way in which to return a neighborlist from
    an atoms object.

    Parameters
    ----------
    self : class
        An ase atoms object to attach feature vector to.

    Returns
    -------
    neighborlist : dict
        The neighbor list attached to the atoms object.
    """
    self._initialize_atoml()
    try:
        return self.atoml['neighborlist']
    except KeyError:
        return None


def set_graph(self, graph):
    """Function to write networkx graph to ase atoms object.

    This function provides a uniform way in which to attach a graph object to
    an atoms object. Can be used in conjunction with the `ase_to_networkx`
    function.

    Parameters
    ----------
    self : class
        An ase atoms object to attach feature vector to.
    graph : object
        The networkx graph object to attach.
    """
    self._initialize_atoml()
    self.atoml['graph'] = graph


def get_graph(self):
    """Function to read networkx graph from ase atoms object.

    This function provides a uniform way in which to return a graph object from
    an atoms object.

    Parameters
    ----------
    self : class
        An ase atoms object to attach feature vector to.

    Returns
    -------
    graph : object
        The networkx graph object attached to the atoms object.
    """
    self._initialize_atoml()
    try:
        return self.atoml['graph']
    except KeyError:
        return None


def _initialize_atoml(self):
    """A base function to initialize the atoml functionality."""
    try:
        self.atoml
    except AttributeError:
        self.atoml = {}
