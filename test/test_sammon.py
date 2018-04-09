import unittest
import numpy as np

from atoml.utilities.sammon import sammons_error

from common import get_data


class TestPrediction(unittest.TestCase):
    """Test out the various prediction routines."""

    def test_rr_loocv(self):
        """Test ridge regression predictions with loocv fitting."""
        train_features, _, _, _ = get_data()

        double = np.concatenate((train_features, train_features), axis=1)

        sammons_error(double, train_features)


if __name__ == '__main__':
    unittest.main()
