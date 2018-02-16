"""Run the full test suite."""
import unittest

from test_suite import ConfigTestCase
from test_scale import TestScaling
from test_predict import TestPrediction
from test_gradients import TestGaussianKernel

if __name__ == '__main__':
    # Add new tests to the following list.
    test_classes_to_run = [ConfigTestCase, TestScaling, TestGaussianKernel,
                           TestPrediction]

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
