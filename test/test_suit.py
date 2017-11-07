"""Run all tests."""
import os
import unittest

from test_data_setup import setup_test
from test_data_clean import outlier_test
import test_feature_optimization as ft
from test_predict import predict_test
from test_predict_scale import scale_test

wkdir = os.getcwd()


class ConfigTestCase(unittest.TestCase):
    """Test suite for AtoML code base."""

    def test_data_setup_func(self):
        """Test data setup routines."""
        setup_test()

    def test_data_clean_func(self):
        """Test data cleaning routines."""
        outlier_test()

    def test_feature_opt_func(self):
        """Test feature optimization routines."""
        train_features, train_targets, test_features = ft.test_extend()
        ft.test_extract(train_features, train_targets, test_features)
        ft.test_screening(train_features, train_targets, test_features)

    def test_predict_func(self):
        """Test prediction routines."""
        predict_test()
        scale_test()


if __name__ == '__main__':
    unittest.main()
    os.remove('{}/fpv_store.sqlite'.format(wkdir))
