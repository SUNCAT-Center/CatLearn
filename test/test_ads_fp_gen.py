"""Script to test data generation functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from ase.build import fcc111, add_adsorbate
from ase.data import atomic_numbers
from atoml.fingerprint.database_adsorbate_api import attach_adsorbate_info
from atoml.fingerprint.periodic_table_data import get_radius
from atoml.fingerprint import FeatureGenerator
wkdir = os.getcwd()


def setup_atoms():
    """Get the atoms objects."""
    symbols = ['Ag', 'Au', 'Cu', 'Pt', 'Pd', 'Ir', 'Rh', 'Ni', 'Co']
    images = []
    for i, s in enumerate(symbols):
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
        atoms.info['dbid'] = i
        images.append(atoms)
    return images


def ads_fp_gen(images):
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
                 gen.primary_surfatom,
                 gen.get_dbid]
    matrix = gen.return_vec(images, train_fpv)
    labels = gen.return_names(train_fpv)
    assert len(labels) == np.shape(matrix)[1]
    if __name__ == '__main__':
        for i, l in enumerate(labels):
            print(i, l)
        for dbid in matrix[:, -1]:
            print('last column:', int(dbid))


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    images = setup_atoms()
    images = attach_adsorbate_info(images)
    ads_fp_gen(images)

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
