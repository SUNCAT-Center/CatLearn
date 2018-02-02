"""Run all tests."""
import os
import unittest

import test_data_setup as ds
import test_scale as st
import test_data_clean as dc
import test_feature_optimization as ft
import test_predict as pt
import test_hierarchy_cv as ht
import test_hypot_scaling as hs
import test_acquisition as ta
import test_io as tio
import test_lml_optimizer as lo
import test_ase_api as taa
from common import get_data

wkdir = os.getcwd()


class ConfigTestCase(unittest.TestCase):
    """Test suite for AtoML code base."""

    def test_data_setup_func(self):
        """Test data setup routines."""
        # Test data setup functions.
        all_cand, data = ds.feature_test()
        ds.cv_test(data)
        ds.db_test(all_cand, data)

        # Test scale data functions.
        train_features, train_targets, test_features, \
            test_targets = st.get_data()
        st.scale_test(train_features, train_targets, test_features)
        st.cluster_test(
            train_features, train_targets, test_features, test_targets)

        # Test api functions.
        taa.ase_api_test()

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

    def test_predict_func(self):
        """Test prediction routines."""
        train_features, train_targets, test_features, \
            test_targets = get_data()
        pt.rr_test(train_features, train_targets, test_features, test_targets)
        pt.gp_test(train_features, train_targets, test_features, test_targets)
        hs.gp_test(train_features, train_targets, test_features, test_targets)

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
