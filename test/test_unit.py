"""Run the full test suite."""
import unittest
import os

# from test_suite import ConfigTestCase
from test_ase_api import TestAPI
from test_feature_base import TestBaseGenerator
from test_feature_generation import TestFeatureGeneration
from test_data_clean import TestDataClean
from test_scale import TestScaling, TestHyperparameterScaling
from test_gradients import TestGaussianKernel
from test_predict import TestPrediction
from test_acquisition import TestAcquisition
from test_io import TestIO

if __name__ == '__main__':
    # Add new tests to the following list.
    test_classes_to_run = [  # ConfigTestCase,
                           TestAPI,
                           TestBaseGenerator,
                           TestFeatureGeneration,
                           TestDataClean,
                           TestScaling,
                           TestHyperparameterScaling,
                           TestGaussianKernel,
                           TestPrediction,
                           TestAcquisition,
                           TestIO]

    # Load in all the unittests.
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    # Compile the test suite.
    suite = unittest.TestSuite(suites_list)

    # Runn all the tests.
    runner = unittest.TextTestRunner()
    results = runner.run(suite)

    os.remove('vec_store.sqlite')
