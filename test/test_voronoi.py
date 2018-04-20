"""Script to test voronoi generation functions."""
from __future__ import print_function
from __future__ import absolute_import

import unittest
import numpy as np

import ase.db

from catlearn.fingerprint.voro_fingerprint import VoronoiFingerprintGenerator


class TestVoronoiFeatures(unittest.TestCase):
    """Test out the adsorbate feature generation."""

    def get_data(self):
        """Get the atoms objects."""
        db = ase.db.connect('data/cubic_perovskites.db')
        atoms = list(db.select(combination='ABO3'))[:10]

        # Compile a list of atoms and target values.
        alist = []
        for row in atoms:
            try:
                alist.append(row.toatoms())
            except AttributeError:
                continue

        return alist

    def test_voronoi_fp_gen(self):
        """Test the feature generation."""
        alist = self.get_data()
        voro = VoronoiFingerprintGenerator(alist)
        features = voro.generate()

        self.assertEqual(np.shape(features)[0], 10)


if __name__ == '__main__':
    unittest.main()
