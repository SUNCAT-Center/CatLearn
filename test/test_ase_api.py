"""Simple tests for the ase api."""
from __future__ import print_function
from __future__ import absolute_import

import os

from ase.ga.data import DataConnection

from atoml.api.ase_api import extend_class

wkdir = os.getcwd()


def ase_api_test():
    """Test the ase api."""
    gadb = DataConnection('{}/data/gadb.db'.format(wkdir))
    all_cand = gadb.get_all_relaxed_candidates()
    extend_class(all_cand[0])
    all_cand[0].store_fp_in_atoms([1., 1., 2., 1.])
    assert all_cand[0].load_fp_in_atoms() == [1., 1., 2., 1.]
    assert all_cand[0].get_chemical_formula() == 'Au92Pt55'


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    ase_api_test()

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
