"""Simple tests for the ase api."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

from ase.ga.data import DataConnection

from atoml.api.ase_api import extend_class
from atoml.fingerprint import StandardFingerprintGenerator

wkdir = os.getcwd()


def ase_api_test():
    """Test the ase api."""
    gadb = DataConnection('{}/data/gadb.db'.format(wkdir))
    all_cand = gadb.get_all_relaxed_candidates()

    cf = all_cand[0].get_chemical_formula()

    extend_class(all_cand[0])
    assert isinstance(all_cand[0], type(all_cand[1]))

    sf = StandardFingerprintGenerator()
    fp = sf.composition_fpv(all_cand[0])
    all_cand[0].set_features(fp)

    assert np.allclose(all_cand[0].get_features(), fp)
    assert all_cand[0].get_chemical_formula() == cf

    extend_class(all_cand[1])
    assert all_cand[1].get_features() is None


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    ase_api_test()

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
