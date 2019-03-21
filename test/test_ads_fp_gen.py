"""Script to test data generation functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import unittest

from ase.build import fcc111, add_adsorbate
from ase.data import atomic_numbers
from ase.constraints import FixAtoms
from catlearn.api.ase_atoms_api import (database_to_list, images_connectivity,
                                        images_pair_distances)
from catlearn.featurize.adsorbate_prep import (autogen_info,
                                               check_reconstructions,
                                               connectivity2ads_index,
                                               termination_info,
                                               z2ads_index, layers2ads_index)
from catlearn.featurize.periodic_table_data import (get_radius,
                                                    default_catlearn_radius)
from catlearn.featurize.slab_utilities import slab_layers
from catlearn.featurize.setup import FeatureGenerator, default_fingerprinters

wkdir = os.getcwd()


class TestAdsorbateFeatures(unittest.TestCase):
    """Test out the adsorbate feature generation."""

    def setup_metals(self, n=None):
        """Get the atoms objects."""
        adsorbates = ['H', 'O', 'C', 'N', 'S', 'Cl', 'P', 'F']
        symbols = ['Ag', 'Au', 'Cu', 'Pt', 'Pd', 'Ir', 'Rh', 'Ni', 'Co']
        if n is not None:
            symbols = symbols[:n]
            adsorbates = adsorbates[:n]
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

    def test_tags(self):
        """Test the feature generation."""
        images = self.setup_metals()
        images = autogen_info(images)
        print(str(len(images)) + ' training examples.')
        gen = FeatureGenerator(nprocs=1)
        train_fpv = default_fingerprinters(gen, 'adsorbates')
        train_fpv += [gen.formal_charges,
                      gen.bag_edges_ads,
                      gen.ads_av,
                      gen.ads_sum]
        matrix = gen.return_vec(images, train_fpv)
        labels = gen.return_names(train_fpv)
        print(np.shape(matrix), type(matrix))
        if __name__ == '__main__':
            for i, l in enumerate(labels):
                print(i, l)
        self.assertTrue(len(labels) == np.shape(matrix)[1])

    def test_constrained_ads(self):
        """Test the feature generation."""
        images = self.setup_metals()
        [atoms.set_tags(np.zeros(len(atoms))) for atoms in images]
        for atoms in images:
            c_atoms = [a.index for a in atoms if
                       a.z < atoms.cell[2, 2] / 2. + 0.1]
            atoms.set_constraint(FixAtoms(c_atoms))
        images = autogen_info(images)
        print(str(len(images)) + ' training examples.')
        gen = FeatureGenerator(nprocs=1)
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
        [atoms.set_tags(np.zeros(len(atoms))) for atoms in images]
        images = autogen_info(images)
        layers2ads_index(images[0],
                         images[0].info['key_value_pairs']['species'])
        print(str(len(images)) + ' training examples.')
        gen = FeatureGenerator(nprocs=1)
        train_fpv = default_fingerprinters(gen, 'adsorbates')
        # Test db specific functions.
        train_fpv += [gen.db_size,
                      gen.ctime,
                      gen.dbid,
                      gen.delta_energy]
        # Old CatApp AxBy fingerprints.
        train_fpv += [gen.catapp_AB]
        matrix = gen.return_vec(images, train_fpv)
        labels = gen.return_names(train_fpv)
        print(np.shape(matrix), type(matrix))
        if __name__ == '__main__':
            for i, l in enumerate(labels):
                print(i, l)
        self.assertTrue(len(labels) == np.shape(matrix)[1])

    def test_recontruction(self):
        images = database_to_list('data/ads_example.db')
        slabs = []
        for atoms in images:
            slab = atoms.copy()
            slab.pop(-1)
            slabs.append(slab)
        images = autogen_info(images)
        slabs = images_connectivity(slabs)
        image_pairs = zip(images, slabs)
        r_index = check_reconstructions(image_pairs)
        for i in range(len(images)):
            species = images[i].info['key_value_pairs']['species']
            connectivity2ads_index(images[i], species)
        self.assertTrue(len(r_index) == 0)

    def test_slab_utils(self):
        images = self.setup_metals(n=2)
        for atoms in images:
            atoms.subsets = {}
            atoms.subsets['ads_atoms'] = \
                z2ads_index(atoms, atoms[-1].symbol)
            slab = atoms[:-1]
            lz, li = slab_layers(slab, 3)
        termination_info(images)

    def test_connectivity(self):
        images = self.setup_metals(n=2)
        images = images_pair_distances(images)
        gen = FeatureGenerator(nprocs=1)
        gen.featurize_atomic_pairs(images)


if __name__ == '__main__':
    unittest.main()
