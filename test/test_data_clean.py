"""Simple tests for data cleaning."""
import numpy as np
import unittest

from atoml.preprocess import clean_data as clean


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
        features = clean.clean_variance(features)['train']

        self.assertTrue(np.shape(features) == (50, 4))

    def test_inf(self):
        """Test cleaning inf variable features."""
        features = np.random.random_sample((50, 5))
        features[1][0] = np.inf
        features = clean.clean_infinite(features)['train']

        self.assertTrue(np.shape(features) == (50, 4))


if __name__ == '__main__':
    unittest.main()
