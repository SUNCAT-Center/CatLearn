import unittest

from test_suite import ConfigTestCase
from test_gradients import TestGaussianKernel

if __name__ == '__main__':
    test_classes_to_run = [ConfigTestCase, TestGaussianKernel]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
