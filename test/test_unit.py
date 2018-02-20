"""Run the full test suite."""
import unittest
import os
import warnings

from test_ase_api import TestAPI
from test_feature_base import TestBaseGenerator
from test_feature_generation import TestFeatureGeneration
from test_ads_fp_gen import TestAdsorbateFeatures
from test_bulk_fp_gen import TestBulkFeatures
from test_data_clean import TestDataClean
from test_feature_optimization import TestFeatureOptimization
from test_scale import TestScaling, TestHyperparameterScaling
from test_gradients import TestGaussianKernel
from test_predict import TestPrediction
from test_acquisition import TestAcquisition
from test_cv import TestCrossValidation
from test_learning_curve import TestCurve
from test_io import TestIO

from test_suite import ConfigTestCase

# Suppress warnings for easier to read output.
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # Add new tests to the following list.
    test_classes_to_run = [TestAPI,
                           TestBaseGenerator,
                           TestFeatureGeneration,
                           TestAdsorbateFeatures,
                           TestBulkFeatures,
                           TestDataClean,
                           TestFeatureOptimization,
                           TestScaling,
                           TestHyperparameterScaling,
                           TestGaussianKernel,
                           TestPrediction,
                           TestAcquisition,
                           TestCrossValidation,
                           TestCurve,
                           TestIO,
                           ConfigTestCase
                           ]

    # Load in all the unittests.
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    # Compile the test suite.
    suite = unittest.TestSuite(suites_list)

    # Run all the tests.
    runner = unittest.TextTestRunner()
    results = runner.run(suite)

    # Clean everything up.
    os.remove('vec_store.sqlite')
    os.remove('hierarchy.pickle')
    os.remove('test.sqlite')
