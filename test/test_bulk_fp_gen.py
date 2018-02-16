"""Script to test data generation functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from ase.build import bulk
from ase.data import atomic_numbers
from atoml.fingerprint.database_adsorbate_api import get_radius
from atoml.fingerprint.setup import return_fpv, get_combined_descriptors
from atoml.fingerprint import BulkFingerprintGenerator
wkdir = os.getcwd()


def setup_metal():
    symbols = ['Ag', 'Au', 'Cu', 'Pt', 'Pd', 'Ir', 'Rh', 'Ni', 'Co']
    images = []
    for s in symbols:
        rs = get_radius(atomic_numbers[s])
        a = 2 * rs * 2 ** 0.5
        atoms = bulk(s, crystalstructure='bcc', a=a)
        images.append(atoms)
    return images


def bulk_fp_gen(images):
    gen = BulkFingerprintGenerator()
    train_fpv = [gen.summation,
                 gen.average,
                 gen.std]
    labels = get_combined_descriptors(train_fpv)
    matrix = return_fpv(images, train_fpv)
    assert len(labels) == np.shape(matrix)[1]


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    images = setup_metal()
    bulk_fp_gen(images)

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
