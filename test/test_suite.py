"""Run all tests."""
import os
import unittest

import test_data_setup as ds
import test_data_clean as dc
import test_feature_optimization as ft
import test_hierarchy_cv as ht
import test_hypot_scaling as hs
import test_acquisition as ta
import test_io as tio
import test_lml_optimizer as lo
import test_ads_fp_gen as afp
from common import get_data

wkdir = os.getcwd()


class ConfigTestCase(unittest.TestCase):
    """Test suite for AtoML code base."""

    def test_ads_fp_funct(self):
        images = afp.setup_atoms()
        images = afp.attach_adsorbate_info(images)
        afp.ads_fg_gen(images)

    def test_data_setup_func(self):
        """Test data setup routines."""
        all_cand, data = ds.feature_test()
        ds.cv_test(data)
        ds.db_test(all_cand, data)

    def test_data_clean_func(self):
        """Test data cleaning routines."""
        dc.outlier_test()
        dc.variance_test()
        dc.inf_test()

    def test_feature_opt_func(self):
        """Test feature optimization routines."""
        ft.test_importance()
        train_features, train_targets, test_features = ft.test_extend()
        ft.test_extract(train_features, train_targets, test_features)
        ft.test_screening(train_features, train_targets, test_features)

    def test_lml_optimizer(self):
        """Test log_marginal_likelihood optimization."""
        train_features, train_targets, test_features, \
            test_targets = get_data()
        lo.lml_test(train_features, train_targets, test_features, test_targets)

    def test_acquisition_func(self):
        """Test acquisition routines."""
        train_features, train_targets, train_atoms, test_features, \
            test_targets, test_atoms = ta.get_data()
        ta.gp_test(train_features, train_targets, train_atoms,
                   test_features, test_targets, test_atoms)

    def test_hierarchy_func(self):
        """Test hierarchy routines."""
        ht.hierarchy_test()

    def test_io_func(self):
        """Test the io routines."""
        train_features, train_targets, test_features, \
            test_targets = get_data()
        model = tio.train_model(train_features, train_targets)
        original = tio.test_model(model, test_features, test_targets)
        tio.test_load(original, test_features, test_targets)
        tio.test_raw(train_features, train_targets, model.regularization,
                     model.kernel_dict)


if __name__ == '__main__':
    unittest.main()
    os.remove('{}/fpv_store.sqlite'.format(wkdir))
