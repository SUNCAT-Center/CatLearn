"""Functions that interface ase with CatLearn."""
import types
import warnings
import numpy as np
import ase.db
from catlearn.utilities.neighborlist import ase_connectivity
from catlearn.featurize.periodic_table_data import default_catlearn_radius
from tqdm import tqdm


def database_to_list(fname, selection=None):
    """Return a list of atoms objects imported from an ase database.

    Parameters
    ----------
    fname : str
        path/filename of ase database.
    selection : list
        search filters to limit the import.
    """
    c = ase.db.connect(fname)
    s = c.select(selection)
    images = []
    for d in s:
        dbid = int(d.id)
        atoms = c.get_atoms(dbid, add_additional_information=True)
        atoms.info['id'] = dbid
        atoms.info['ctime'] = float(d.ctime)
        atoms.subsets = {}
        if hasattr(d, 'data') and 'connectivity' in dict(d.data):
            atoms.connectivity = np.array(d.data.connectivity)
        images.append(atoms)

    return images


def images_connectivity(images, check_cn_max=False):
    """Return a list of atoms objects imported from an ase database.

    Parameters
    ----------
    fname : str
        path/filename of ase database.
    selection : list
        search filters to limit the import.
    """
    for atoms in tqdm(images):
        if not hasattr(atoms, 'connectivity'):
            radii = [default_catlearn_radius(z) for z in atoms.numbers]
            atoms.connectivity = ase_connectivity(atoms, cutoffs=radii)
            if check_cn_max:
                n_connections = np.sum(atoms.connectivity, axis=0)
                if max(n_connections) > 12:
                    msg = 'atom has more than 12 connections.'
                    if 'id' in atoms.info:
                        msg += str(atoms.info['id'])
                    warnings.warn(msg)
    return images


def images_pair_distances(images, mic=True):
    """Return a list of atoms objects imported from an ase database.

    Parameters
    ----------
    fname : str
        path/filename of ase database.
    selection : list
        search filters to limit the import.
    """
    for atoms in tqdm(images):
        atoms.pair_distances = np.array(atoms.get_all_distances(mic=mic))
    return images


def extend_atoms_class(atoms):
    """A wrapper to add extra functionality to ase atoms objects.

    Parameters
    ----------
    atoms : class
        An ase atoms object.
    """
    atoms._initialize_catlearn = types.MethodType(_initialize_catlearn, atoms)
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
    self._initialize_catlearn()
    self.catlearn['features'] = fp


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
    self._initialize_catlearn()
    try:
        return self.catlearn['features']
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
    self._initialize_catlearn()
    self.catlearn['neighborlist'] = neighborlist


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
    self._initialize_catlearn()
    try:
        return self.catlearn['neighborlist']
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
    self._initialize_catlearn()
    self.catlearn['graph'] = graph


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
    self._initialize_catlearn()
    try:
        return self.catlearn['graph']
    except KeyError:
        return None


def _initialize_catlearn(self):
    """A base function to initialize the catlearn functionality."""
    try:
        self.catlearn
    except AttributeError:
        self.catlearn = {}
