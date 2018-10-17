"""Simple tests for the ase api."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import unittest

from ase.ga.data import DataConnection

from catlearn.api.ase_atoms_api import extend_atoms_class
from catlearn.api.catmap import catmap_energy_landscape
from catlearn.api.networkx_graph_api import (ase_to_networkx,
                                             networkx_to_adjacency)
from catlearn.featurize.setup import FeatureGenerator

wkdir = os.getcwd()


class TestEnergyLandscape(object):
    def __init__(self):
        self.formation_energies = {}
        self.dbid = {}
        self.std = {}

    def _get_adsorbate_fields(self, d):
        fields = [1, str(d.species), str(d.name), str(d.crystal),
                  str(d.surf_lattice), str(d.facet), '2x2x3']
        return fields


class TestAPI(unittest.TestCase):
    """Test out the ASE api."""

    def test_networkx_api(self):
        """Test the ase api."""
        gadb = DataConnection('{}/data/gadb.db'.format(wkdir))
        all_cand = gadb.get_all_relaxed_candidates()
        g = ase_to_networkx(all_cand[1])

        self.assertEqual(len(g), len(all_cand[1]))

        matrix = networkx_to_adjacency(g)
        self.assertEqual(np.shape(matrix),
                         (len(all_cand[1]), len(all_cand[1])))

    def test_ase_api(self):
        """Test the ase api."""
        gadb = DataConnection('{}/data/gadb.db'.format(wkdir))
        all_cand = gadb.get_all_relaxed_candidates()

        cf = all_cand[0].get_chemical_formula()

        extend_atoms_class(all_cand[0])
        self.assertTrue(isinstance(all_cand[0], type(all_cand[1])))

        f = FeatureGenerator()
        fp = f.composition_vec(all_cand[0])
        all_cand[0].set_features(fp)

        self.assertTrue(np.allclose(all_cand[0].get_features(), fp))
        self.assertTrue(all_cand[0].get_chemical_formula() == cf)

        extend_atoms_class(all_cand[1])
        self.assertTrue(all_cand[1].get_features() is None)

        g = ase_to_networkx(all_cand[2])
        all_cand[2].set_graph(g)

        self.assertTrue(all_cand[2].get_graph() == g)
        self.assertTrue(all_cand[1].get_graph() is None)

    def test_catmap_api(self):
        fname = 'data/ads_example.db'
        database_ids = [1, 2]
        prediction = [0.1, 0.2]
        uncertainty = [0.2, 0.1]
        energy_landscape = TestEnergyLandscape()
        energy_landscape = catmap_energy_landscape(fname, database_ids,
                                                   prediction,
                                                   uncertainty,
                                                   catmap=energy_landscape)
        if __name__ == '__main__':
            print(energy_landscape.formation_energies)


if __name__ == '__main__':
    unittest.main()
