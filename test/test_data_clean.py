"""Simple tests for data cleaning."""
import numpy as np
import unittest

from catlearn.preprocess import clean_data as clean
from catlearn.setup.general_preprocess import GeneralPrepreprocess

from common import get_data


class TestDataClean(unittest.TestCase):
    """Test out the data cleaning functions."""

    def test_outlier(self):
        """Test outlier removal from toy features."""
        f = np.arange(200).reshape(50, 4)

        t = np.random.random_sample((50,))
        t[2] += 200000.
        t[0] -= 200000.

        d = clean.remove_outliers(features=f, targets=t, constraint=None)

        self.assertTrue(np.shape(d['features']) != np.shape(f))
        self.assertTrue(np.shape(d['targets']) != np.shape(t))
        self.assertTrue(np.shape(d['features'])[0] ==
                        np.shape(d['targets'])[0])

    def test_variance(self):
        """Test cleaning zero variace features."""
        features = np.random.random_sample((50, 5))
        features[:, 1:2] = 109.982
        test = np.random.random_sample((100, 5))
        labels = 5 * ['test_label']

        informative = clean.clean_variance(features, test=test, labels=labels)

        self.assertTrue(np.shape(informative['train']) == (50, 4))
        self.assertTrue(np.shape(informative['test']) == (100, 4))

    def test_inf(self):
        """Test cleaning inf variable features."""
        features = np.random.random_sample((50, 5))
        features[0, 0] = np.nan
        features[40:, 1] = np.nan
        test = np.random.random_sample((100, 5))
        labels = 5 * ['test_label']

        finite = clean.clean_infinite(features, test=test, labels=labels,
                                      max_impute_fraction=0.1)

        self.assertTrue(np.shape(finite['train']) == (50, 4))
        self.assertTrue(np.shape(finite['test']) == (100, 4))

    def test_general(self):
        """Test the general cleaning/scaling function."""
        train_features, train_targets, test_features, _ = get_data()

        clean = GeneralPrepreprocess()
        clean_train, clean_targets, clean_test = clean.process(
            train_features, train_targets, test_features)

        self.assertNotEqual(np.shape(train_features), np.shape(clean_train))
        self.assertEqual(np.shape(train_targets), np.shape(clean_targets))

        transform_test = clean.transform(test_features)

        self.assertTrue(np.allclose(clean_test, transform_test))


if __name__ == '__main__':
    unittest.main()
