"""Run the full test suite."""
import unittest
import os
import warnings

from test_1_feature_generation import TestFeatureGeneration
from test_api import TestAPI
from test_feature_base import TestBaseGenerator
from test_ads_fp_gen import TestAdsorbateFeatures
from test_bulk_fp_gen import TestBulkFeatures
from test_voronoi import TestVoronoiFeatures
from test_neighborlist import TestNeighborList
from test_data_clean import TestDataClean
from test_feature_optimization import TestFeatureOptimization
from test_scale import TestScaling, TestHyperparameterScaling
from test_gradients import TestGaussianKernel
from test_predict import TestPrediction
from test_acquisition import TestAcquisition
from test_validation import TestValidation
from test_learning_curve import TestCurve
from test_io import TestIO
from test_ga import TestGeneticAlgorithm
from test_autocorrelation import TestAutoCorrelation

from test_functions import ConfigTestCase

# Suppress warnings for easier to read output.
warnings.filterwarnings("ignore")


def setup_suite(class_list):
    """Basic function to setup unittest test suite."""
    # Load in all the unittests.
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None

    suites_list = []
    for test_class in class_list:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    # Compile the test suite.
    suite = unittest.TestSuite(suites_list)

    # Run all the tests.
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    # Add other tests to the following list.
    setup_suite([
        TestFeatureGeneration,
        TestAPI,
        TestBaseGenerator,
        TestAdsorbateFeatures,
        TestBulkFeatures,
        TestVoronoiFeatures,
        TestNeighborList,
        TestDataClean,
        TestFeatureOptimization,
        TestScaling,
        TestHyperparameterScaling,
        TestGaussianKernel,
        TestPrediction,
        TestAcquisition,
        TestValidation,
        TestCurve,
        TestIO,
        TestGeneticAlgorithm,
        ConfigTestCase,
        TestAutoCorrelation
    ])

    # Clean everything up.
    os.remove('vec_store.sqlite')
    os.remove('hierarchy.pickle')
    os.remove('test.sqlite')
    os.remove('cvsave.pickle')
    os.remove('cvsave.json')
    os.remove('gaWrite.json')
