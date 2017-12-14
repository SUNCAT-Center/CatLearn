"""Run all tests."""
import os
import unittest

import test_data_setup as ds
from test_scale import st
import test_data_clean as dc
import test_feature_optimization as ft
import test_predict as pt
import test_hierarchy_cv as ht

wkdir = os.getcwd()


class ConfigTestCase(unittest.TestCase):
    """Test suite for AtoML code base."""

    def test_data_setup_func(self):
        """Test data setup routines."""
        all_cand, data = ds.feature_test()
        ds.cv_test(data)
        ds.db_test(all_cand, data)
        st.scale_test()

    def test_data_clean_func(self):
        """Test data cleaning routines."""
        dc.outlier_test()
        dc.variance_test()
        dc.inf_test()

    def test_feature_opt_func(self):
        """Test feature optimization routines."""
        train_features, train_targets, test_features = ft.test_extend()
        ft.test_extract(train_features, train_targets, test_features)
        ft.test_screening(train_features, train_targets, test_features)

    def test_predict_func(self):
        """Test prediction routines."""
        train_features, train_targets, test_features, \
            test_targets = pt.get_data()
        pt.rr_test(train_features, train_targets, test_features, test_targets)
        pt.gp_test(train_features, train_targets, test_features, test_targets)

    def test_hierarchy_func(self):
        """Test hierarchy routines."""
        ht.hierarchy_test()


if __name__ == '__main__':
    unittest.main()
    os.remove('{}/fpv_store.sqlite'.format(wkdir))
