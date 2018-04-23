"""Simple tests for the ase api."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import unittest

from ase.ga.data import DataConnection

from catlearn.api.ase_atoms_api import extend_atoms_class
from catlearn.api.networkx_graph_api import (ase_to_networkx,
                                             networkx_to_adjacency)
from catlearn.fingerprint.setup import FeatureGenerator

wkdir = os.getcwd()


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


if __name__ == '__main__':
    unittest.main()
