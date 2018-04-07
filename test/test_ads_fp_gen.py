"""Script to test data generation functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import unittest

from ase.build import fcc111, add_adsorbate
from ase.data import atomic_numbers
from ase.constraints import FixAtoms
from atoml.fingerprint.adsorbate_prep import autogen_info
from atoml.fingerprint.periodic_table_data import (get_radius,
                                                   default_atoml_radius)
from atoml.fingerprint.setup import FeatureGenerator

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
                c_atoms = [a.index for a in atoms if
                       a.z < atoms.cell[2, 2]/2. + 0.1]
                atoms.set_constraint(FixAtoms(c_atoms))
                h = (default_atoml_radius(atomic_numbers[ads]) + rs) / 2 ** 0.5
                add_adsorbate(atoms, ads, h, 'bridge')
                images.append(atoms)
        return images

    def test_ads_fp_gen(self):
        """Test the feature generation."""
        images = self.setup_atoms()
        images = autogen_info(images)
        print(str(len(images)) + ' training examples.')
        gen = FeatureGenerator()
        train_fpv = [gen.mean_chemisorbed_atoms,
                     gen.count_chemisorbed_fragment,
                     gen.count_ads_atoms,
                     gen.count_ads_bonds,
                     gen.mean_site,
                     gen.sum_site,
                     gen.mean_surf_ligands,
                     gen.term,
                     gen.bulk,
                     gen.strain,
                     gen.en_difference,
                     # gen.ads_av,
                     # gen.ads_sum,
                     ]
        matrix = gen.return_vec(images, train_fpv)
        labels = gen.return_names(train_fpv)
        print(np.shape(matrix), type(matrix))
        if __name__ == '__main__':
            for i, l in enumerate(labels):
                print(i, l)
        self.assertTrue(len(labels) == np.shape(matrix)[1])


if __name__ == '__main__':
    unittest.main()
