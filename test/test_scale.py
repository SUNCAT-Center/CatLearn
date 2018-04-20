"""Script to test the scaling functions."""
from __future__ import print_function
from __future__ import absolute_import

import unittest
import os
import numpy as np

from catlearn.utilities import DescriptorDatabase
from catlearn.preprocess.scaling import (
    standardize, normalize, min_max, unit_length, target_standardize,
    target_normalize, target_center)
from catlearn.utilities.clustering import cluster_features
from catlearn.regression import GaussianProcess
from catlearn.regression.gpfunctions import hyperparameter_scaling as hs

from common import get_data

wkdir = os.getcwd()

train_size, test_size = 45, 5


class TestScaling(unittest.TestCase):
    """Test scaling routines."""

    def get_data(self):
        """Simple function to pull some training and test data."""
        # Attach the database.
        dd = DescriptorDatabase(db_name='{}/vec_store.sqlite'.format(wkdir),
                                table='FingerVector')

        # Pull the features and targets from the database.
        names = dd.get_column_names()
        features, targets = names[1:-1], names[-1:]
        feature_data = dd.query_db(names=features)
        target_data = np.reshape(dd.query_db(names=targets),
                                 (np.shape(feature_data)[0], ))

        # Split the data into so test and training sets.
        train_features = feature_data[:train_size, :]
        train_targets = target_data[:train_size]
        test_features = feature_data[test_size:, :]
        test_targets = target_data[test_size:]

        return train_features, train_targets, test_features, test_targets

    def test_scale(self):
        """Test data scaling functions."""
        train_features, train_targets, test_features, _ = self.get_data()
        sfp = standardize(train_matrix=train_features,
                          test_matrix=test_features)
        sfpg = standardize(train_matrix=train_features,
                           test_matrix=test_features, local=False)
        self.assertFalse(np.allclose(sfp['train'], sfpg['train']))

        nfp = normalize(train_matrix=train_features, test_matrix=test_features)
        nfpg = normalize(train_matrix=train_features,
                         test_matrix=test_features, local=False)
        self.assertFalse(np.allclose(nfp['train'], nfpg['train']))

        mmfp = min_max(train_matrix=train_features, test_matrix=test_features)
        mmfpg = min_max(train_matrix=train_features, test_matrix=test_features,
                        local=False)
        self.assertFalse(np.allclose(mmfp['train'], mmfpg['train']))

        ulfp = unit_length(train_matrix=train_features,
                           test_matrix=test_features)
        ulfpg = unit_length(train_matrix=train_features,
                            test_matrix=test_features, local=False)
        self.assertTrue(np.allclose(ulfp['train'], ulfpg['train']))

        ts = target_standardize(train_targets)
        self.assertFalse(np.allclose(ts['target'], train_targets))

        ts = target_normalize(train_targets)
        self.assertFalse(np.allclose(ts['target'], train_targets))

        ts = target_center(train_targets)
        self.assertFalse(np.allclose(ts['target'], train_targets))

    def test_cluster(self):
        """Test clustering function."""
        train_features, train_targets, test_features, \
            test_targets = self.get_data()
        cf = cluster_features(
            train_matrix=train_features, train_target=train_targets,
            test_matrix=test_features, test_target=test_targets, k=2)
        self.assertTrue(len(cf['train_features']) == 2)
        self.assertTrue(len(cf['train_target']) == 2)
        self.assertTrue(len(cf['test_features']) == 2)
        self.assertTrue(len(cf['test_target']) == 2)


class TestHyperparameterScaling(unittest.TestCase):
    """Test rescaling of hyperparameters."""

    def test_gp(self):
        """Test Gaussian process predictions with rescaling."""
        train_features, train_targets, test_features, test_targets = get_data()
        # Test prediction routine with gaussian kernel.
        kdict = {'k1': {'type': 'gaussian', 'width': 1., 'scaling': 1.}}
        gp = GaussianProcess(
            train_fp=train_features, train_target=train_targets,
            kernel_dict=kdict, regularization=1e-3,
            optimize_hyperparameters=True, scale_data=True)
        pred = gp.predict(test_fp=test_features,
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True,
                          uncertainty=True)

        opt = gp.kernel_dict['k1']['width']
        orig = hs.rescale_hyperparameters(gp.scaling,
                                          gp.kernel_dict)['k1']['width']
        assert not np.allclose(opt, orig)
        scaled = hs.hyperparameters(gp.scaling, gp.kernel_dict)['k1']['width']
        assert np.allclose(opt, scaled)

        print('gaussian prediction (rmse):',
              pred['validation_error']['rmse_average'])


if __name__ == '__main__':
    unittest.main()
