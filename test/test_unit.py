"""Run the full test suite."""
import unittest
import os

# from test_suite import ConfigTestCase
from test_data_setup import TestFeatureGeneration
from test_scale import TestScaling, TestHyperparameterScaling
from test_gradients import TestGaussianKernel
from test_predict import TestPrediction
from test_io import TestIO

if __name__ == '__main__':
    # Add new tests to the following list.
    test_classes_to_run = [  # ConfigTestCase,
                           TestFeatureGeneration,
                           TestScaling,
                           TestHyperparameterScaling,
                           TestGaussianKernel,
                           TestPrediction,
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
