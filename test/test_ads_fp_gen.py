"""Script to test data generation functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from ase.build import fcc111, add_adsorbate
from ase.data import atomic_numbers
from atoml.fingerprint.database_adsorbate_api import (get_radius,
                                                      attach_adsorbate_info)
from atoml.fingerprint import FeatureGenerator
wkdir = os.getcwd()


def setup_atoms():
    """Get the atoms objects."""
    symbols = ['Ag', 'Au', 'Cu', 'Pt', 'Pd', 'Ir', 'Rh', 'Ni', 'Co']
    images = []
    for s in symbols:
        rs = get_radius(atomic_numbers[s])
        a = 2 * rs * 2 ** 0.5
        atoms = fcc111(s, (2, 2, 3), a=a)
        atoms.center(vacuum=6, axis=2)
        h = (get_radius(6) + rs) / 2 ** 0.5
        add_adsorbate(atoms, 'C', h, 'bridge')
        atoms.info['species'] = 'C'
        atoms.info['layers'] = 3
        atoms.info['bulk'] = s
        atoms.info['termination'] = s
        images.append(atoms)
    return images


def ads_fg_gen(images):
    """Test the feature generation."""
    gen = FeatureGenerator()
    train_fpv = [gen.ads_nbonds,
                 gen.primary_addatom,
                 gen.primary_adds_nn,
                 gen.bulk,
                 gen.term,
                 gen.Z_add,
                 gen.ads_av,
                 gen.primary_surf_nn,
                 gen.primary_surfatom]
    matrix = gen.return_vec(images, train_fpv)
    labels = gen.return_names(train_fpv)
    assert len(labels) == np.shape(matrix)[1]


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    images = setup_atoms()
    images = attach_adsorbate_info(images)
    ads_fg_gen(images)

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
