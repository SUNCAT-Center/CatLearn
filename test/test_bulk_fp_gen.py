"""Script to test data generation functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import unittest

from ase.build import bulk
from ase.data import atomic_numbers

from catlearn.fingerprint.periodic_table_data import get_radius
from catlearn.fingerprint.setup import FeatureGenerator, default_fingerprinters

wkdir = os.getcwd()


class TestBulkFeatures(unittest.TestCase):
    """Test out the adsorbate feature generation."""

    def setup_metal(self):
        """Get the atoms objects."""
        symbols = ['Ag', 'Au', 'Cu', 'Pt', 'Pd', 'Ir', 'Rh', 'Ni', 'Co']
        images = []
        for s in symbols:
            rs = get_radius(atomic_numbers[s])
            a = 2 * rs * 2 ** 0.5
            atoms = bulk(s, crystalstructure='bcc', a=a)
            images.append(atoms)
        return images

    def test_bulk_fp_gen(self):
        """Test the feature generation."""
        images = self.setup_metal()

        gen = FeatureGenerator()
        train_fpv = default_fingerprinters(gen, 'bulk')
        matrix = gen.return_vec(images, train_fpv)
        labels = gen.return_names(train_fpv)
        print(np.shape(matrix), print(type(matrix)))
        self.assertTrue(len(labels) == np.shape(matrix)[1])


if __name__ == '__main__':
    unittest.main()
