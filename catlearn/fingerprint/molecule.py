"""Functions to build a gas phase molecule fingerprint."""
from catlearn.featurize.base import BaseGenerator
from catlearn.featurize.periodic_table_data import list_mendeleev_params
import networkx as nx
import numpy as np


default_parameters = [
    'atomic_number',
    'covalent_radius_cordero',
    'en_pauling']


class AutoCorrelationFingerprintGenerator(BaseGenerator):
    """Class for constructing an autocorrelation fingerprint."""

    def __init__(self, **kwargs):
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
        # Slab periodic table parameters.
        if not hasattr(self, 'dstar'):
            self.dstar = kwargs.get('dstar')

        if self.dstar is None:
            self.dstar = 2

        if not hasattr(self, 'parameters'):
            self.parameters = kwargs.get('parameters')

        if self.parameters is None:
            self.parameters = default_parameters

        super(AutoCorrelationFingerprintGenerator, self).__init__(**kwargs)

    def get_autocorrelation(self, atoms):
        """Return the autocorrelation fingerprint for a molecule."""
        connectivity = atoms.connectivity

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
