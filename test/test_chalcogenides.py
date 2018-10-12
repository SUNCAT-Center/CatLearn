"""Script to test data generation functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import unittest

from catlearn.api.ase_atoms_api import database_to_list, images_connectivity
from catlearn.featurize.adsorbate_prep import (slab_positions2ads_index,
                                               slab_index,
                                               attach_cations,
                                               info2primary_index)
from catlearn.featurize.periodic_table_data import stat_mendeleev_params
from catlearn.featurize.setup import FeatureGenerator, default_fingerprinters
from catlearn.utilities.utilities import formal_charges


wkdir = os.getcwd()


class TestChalcogenides(unittest.TestCase):
    """Test out the adsorbate feature generation."""

    def test_adsorption(self):
        images = database_to_list('data/bajdichWO32018_ads.db')
        images = images_connectivity(images)
        slabs = database_to_list('data/bajdichWO32018_slabs.db')
        slabs_dict = {}
        for slab in slabs:
            slabs_dict[slab.info['id']] = slab
        for i, atoms in enumerate(images):
            species = atoms.info['key_value_pairs']['species']
            atoms.subsets['ads_atoms'] = \
                slab_positions2ads_index(atoms, slabs[i], species)
            if 'slab_atoms' not in atoms.subsets:
                atoms.subsets['slab_atoms'] = slab_index(atoms)
            if ('chemisorbed_atoms' not in atoms.subsets or
                'site_atoms' not in atoms.subsets or
                    'ligand_atoms' not in atoms.subsets):
                chemi, site, ligand = info2primary_index(atoms)
                atoms.subsets['chemisorbed_atoms'] = chemi
                atoms.subsets['site_atoms'] = site
                atoms.subsets['ligand_atoms'] = ligand
            attach_cations(atoms, anion_number=8)
            charges = formal_charges(atoms)
            atoms.set_initial_charges(charges)
        gen = FeatureGenerator(nprocs=1)
        train_fpv = default_fingerprinters(gen, 'chalcogenides')
        matrix = gen.return_vec(images, train_fpv)
        labels = gen.return_names(train_fpv)
        if __name__ == '__main__':
            for i, l in enumerate(labels):
                print(i, l)
            print(atoms.get_initial_charges())
        self.assertTrue(len(labels) == np.shape(matrix)[1])

    def test_periodic_table(self):
        r, w = stat_mendeleev_params('MoS2', params=None)
        self.assertTrue(len(w) == np.shape(r)[0])


if __name__ == '__main__':
    unittest.main()
