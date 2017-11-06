"""Run all the valid tests."""
import os
import unittest

from test_data_setup import setup_test
from test_data_clean import outlier_test
from test_feature_optimization import feature_test
from test_predict import predict_test
from test_predict_scale import scale_test


class ConfigTestCase(unittest.TestCase):
    def test_data_clean_func(self):
        outlier_test()

    def test_data_setup_func(self):
        setup_test()

    def test_feature_opt_func(self):
        feature_test()

    def test_predict_func(self):
        predict_test()
        scale_test()


if __name__ == '__main__':
    unittest.main()
    os.remove('fpv_store.sqlite')
