"""Script to test data generation functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import unittest

from ase.build import fcc111, add_adsorbate
from ase.data import atomic_numbers
from ase.constraints import FixAtoms
from catlearn.api.ase_atoms_api import database_to_list, images_connectivity
from catlearn.fingerprint.adsorbate_prep import (autogen_info,
                                                 check_reconstructions,
                                                 connectivity2ads_index,
                                                 slab_positions2ads_index,
                                                 slab_index,
                                                 attach_cations,
                                                 info2primary_index)
from catlearn.fingerprint.periodic_table_data import (get_radius,
                                                      default_catlearn_radius,
                                                      stat_mendeleev_params)
from catlearn.fingerprint.setup import FeatureGenerator, default_fingerprinters

wkdir = os.getcwd()


class TestAdsorbateFeatures(unittest.TestCase):
    """Test out the adsorbate feature generation."""

    def setup_metals(self):
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

    def test_tags(self):
        """Test the feature generation."""
        images = self.setup_metals()
        images = autogen_info(images)
        print(str(len(images)) + ' training examples.')
        gen = FeatureGenerator(nprocs=1)
        train_fpv = default_fingerprinters(gen, 'adsorbates')
        train_fpv += [gen.formal_charges,
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
        reconstructed = check_reconstructions(image_pairs)
        for i in range(len(images)):
            species = images[i].info['key_value_pairs']['species']
            connectivity2ads_index(images[i], species)
        self.assertTrue(len(reconstructed) == 0)

    def test_chalcogenides(self):
        images = database_to_list('data/bajdichWO32018_ads.db')
        images = images_connectivity(images)
        slabs = database_to_list('data/bajdichWO32018_slabs.db')
        slabs_dict = {}
        for slab in slabs:
            slabs_dict[slab.info['id']] = slab
        for i in range(len(images)):
            species = images[i].info['key_value_pairs']['species']
            images[i].subsets['ads_atoms'] = \
                slab_positions2ads_index(images[i], slabs[i], species)
            if 'slab_atoms' not in images[i].subsets:
                images[i].subsets['slab_atoms'] = slab_index(images[i])
            if ('chemisorbed_atoms' not in images[i].subsets or
                'site_atoms' not in images[i].subsets or
                    'ligand_atoms' not in images[i].subsets):
                chemi, site, ligand = info2primary_index(images[i])
                images[i].subsets['chemisorbed_atoms'] = chemi
                images[i].subsets['site_atoms'] = site
                images[i].subsets['ligand_atoms'] = ligand
            attach_cations(images[i], anion_number=8)
        gen = FeatureGenerator(nprocs=1)
        train_fpv = default_fingerprinters(gen, 'chalcogenides')
        matrix = gen.return_vec(images, train_fpv)
        labels = gen.return_names(train_fpv)
        if __name__ == '__main__':
            for i, l in enumerate(labels):
                print(i, l)
        self.assertTrue(len(labels) == np.shape(matrix)[1])

    def test_periodic_table(self):
        r, w = stat_mendeleev_params('MoS2', params=None)
        self.assertTrue(len(w) == np.shape(r)[0])


if __name__ == '__main__':
    unittest.main()
