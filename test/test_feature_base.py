"""Simple tests for the base feature generator."""
from __future__ import print_function
from __future__ import absolute_import

from ase.ga.data import DataConnection

from atoml import __path__ as atoml_path
from atoml.fingerprint.base import FeatureGenerator

atoml_path = '/'.join(atoml_path[0].split('/')[:-1])


def feature_base_test():
    """Test the base feature generator."""
    gadb = DataConnection('{}/data/gadb.db'.format(atoml_path))
    all_cand = gadb.get_all_relaxed_candidates()

    f = FeatureGenerator()
    nl = f.atoms_neighborlist(all_cand[0])
    assert f.get_neighborlist(all_cand[0]) == nl


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    feature_base_test()

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
