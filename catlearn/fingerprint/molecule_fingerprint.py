"""Functions to build a gas phase molecule fingerprint."""
from catlearn.utilities.neighborlist import catlearn_neighborlist
from catlearn.fingerprint.periodic_table_data import list_mendeleev_params
import networkx as nx
import numpy as np
from ase import Atoms

default_parameters = [
    'atomic_number',
    'covalent_radius_cordero',
    'en_pauling',
]


class AutoCorrelationFingerprintGenerator():
    """Class for constructing an autocorrelation fingerprint."""

    def __init__(self, images, dstar=0, parameters=None):
        """Initialize.

        Parameters
        ----------
        images : list of objects (n,)
            Atoms objects to create fingerprints for.
        dstar : int
            Maximum distance to consider for autocorrelation.
        parameters : list
            Parameters to use for the autocorrelation
        """
        if isinstance(images, Atoms):
            images = [images]

        self.images = images
        self.dstar = dstar

        if parameters is None:
            self.parameters = default_parameters

    def generate(self):
        """Return an (n, m) array of fingerprints."""
        fp_length = len(self.parameters) * (self.dstar + 1)
        fingerprints = np.zeros((len(self.images), fp_length))

        for i, atoms in enumerate(self.images):
            fingerprints[i] = self.get_autocorrelation(atoms)

        return fingerprints

    def get_autocorrelation(self, atoms):
        """Return the autocorrelation fingerprint for a molecule."""
        connectivity = catlearn_neighborlist(atoms)

        G = nx.Graph(connectivity)
        distance_matrix = nx.floyd_warshall_numpy(G)
        Bm = np.zeros(distance_matrix.shape)

        n = len(self.parameters)
        W = list_mendeleev_params(atoms.numbers, self.parameters).T

        fingerprint = np.zeros(n * (self.dstar + 1))
        for dd in range(self.dstar + 1):
            B = Bm.copy()
            B[distance_matrix == dd] = 1
            AC = np.dot(np.dot(W, B), W.T).diagonal()
            fingerprint[n * dd:n * (dd + 1)] = AC

        return fingerprint
