"""Script to test data generation functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import unittest

from ase.build import fcc111, add_adsorbate
from ase.data import atomic_numbers
from ase.constraints import FixAtoms
from catlearn.api.ase_atoms_api import database_to_list
from catlearn.fingerprint.adsorbate_prep import autogen_info
from catlearn.fingerprint.periodic_table_data import (get_radius,
                                                      default_catlearn_radius)
from catlearn.fingerprint.setup import FeatureGenerator, default_fingerprinters

wkdir = os.getcwd()


class TestAdsorbateFeatures(unittest.TestCase):
    """Test out the adsorbate feature generation."""

    def setup_atoms(self):
        """Get the atoms objects."""
        adsorbates = ['H', 'O', 'C', 'N', 'S', 'Cl', 'P', 'F']
        symbols = ['Ag', 'Au', 'Cu', 'Pt', 'Pd', 'Ir', 'Rh', 'Ni', 'Co']
        images = []
        for i, s in enumerate(symbols):
            rs = get_radius(atomic_numbers[s])
            a = 2 * rs * 2 ** 0.5
            for ads in adsorbates:
                atoms = fcc111(s, (2, 2, 3), a=a)
                atoms.center(vacuum=6, axis=2)
                h = (default_catlearn_radius(
                    atomic_numbers[ads]) + rs) / 2 ** 0.5
                add_adsorbate(atoms, ads, h, 'bridge')
                images.append(atoms)
        return images

    def test_raw_ads(self):
        """Test the feature generation."""
        images = self.setup_atoms()
        images = autogen_info(images)
        print(str(len(images)) + ' training examples.')
        gen = FeatureGenerator(nprocs=None)
        train_fpv = default_fingerprinters(gen, 'adsorbates')
        matrix = gen.return_vec(images, train_fpv)
        labels = gen.return_names(train_fpv)
        print(np.shape(matrix), type(matrix))
        if __name__ == '__main__':
            for i, l in enumerate(labels):
                print(i, l)
        self.assertTrue(len(labels) == np.shape(matrix)[1])

    def test_constrained_ads(self):
        """Test the feature generation."""
        images = self.setup_atoms()
        for atoms in images:
            c_atoms = [a.index for a in atoms if
                       a.z < atoms.cell[2, 2] / 2. + 0.1]
            atoms.set_constraint(FixAtoms(c_atoms))
        images = autogen_info(images)
        print(str(len(images)) + ' training examples.')
        gen = FeatureGenerator(nprocs=None)
        train_fpv = default_fingerprinters(gen, 'adsorbates')
        matrix = gen.return_vec(images, train_fpv)
        labels = gen.return_names(train_fpv)
        print(np.shape(matrix), type(matrix))
        if __name__ == '__main__':
            for i, l in enumerate(labels):
                print(i, l)
        self.assertTrue(len(labels) == np.shape(matrix)[1])

    def test_db_ads(self):
        """Test the feature generation."""
        images = database_to_list('data/ads_example.db')
        images = autogen_info(images)
        print(str(len(images)) + ' training examples.')
        gen = FeatureGenerator(nprocs=1)
        train_fpv = default_fingerprinters(gen, 'adsorbates')
        train_fpv += [gen.db_size,
                      gen.ctime,
                      gen.dbid,
                      gen.delta_energy]
        matrix = gen.return_vec(images, train_fpv)
        labels = gen.return_names(train_fpv)
        print(np.shape(matrix), type(matrix))
        if __name__ == '__main__':
            for i, l in enumerate(labels):
                print(i, l)
        self.assertTrue(len(labels) == np.shape(matrix)[1])


if __name__ == '__main__':
    unittest.main()
