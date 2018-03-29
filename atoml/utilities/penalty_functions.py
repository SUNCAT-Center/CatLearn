"""Class with penalty functions."""
import numpy as np
from scipy.spatial import distance


class PenaltyFunctions(object):
    """Base class for penalty functions."""

    def __init__(self, targets=None, predictions=None, uncertainty=None,
                 train_features=None, test_features=None):
        """Initialization of class.

        Parameters
        ----------
        targets : list
            List of known target values.
        predictions : list
            List of predictions from the GP.
        uncertainty : list
            List of variance on the GP predictions.
        train_features : array
            Feature matrix for the training data.
        test_features : array
            Feature matrix for the test data.

        """
        self.targets = targets
        self.predictions = predictions
        self.uncertainty = uncertainty
        self.train_features = train_features
        self.test_features = test_features

    def penalty_close(self, c_min_crit=1e5, d_min_crit=1e-5):
        """Penalize data that is too close.

        Pass an array of test features and train features and returns an array
        of penalties due to 'too short distance' ensuring no duplicates are
        added.

        Parameters
        ----------
        d_min_crit : float
            Critical distance.
        c_min_crit : float
            Constant for penalty minimum distance.
        penalty_min: array
            Array containing the penalty to add.
        """
        penalty_min = []
        for i in self.test_features:
            d_min = np.min(
                distance.cdist([i], self.train_features, 'euclidean'))
            p = 0.0
            if d_min < d_min_crit:
                p = c_min_crit * (d_min - d_min_crit)**2
            penalty_min.append(p)

        return penalty_min

    def penalty_far(self, c_max_crit=1e2, d_max_crit=10.0):
        """Penalize data that is too far.

        Pass an array of test features and train features and returns an array
        of penalties due to 'too far distance'. This prevents to explore
        configurations that are unrealistic.

        Parameters
        ----------
        d_max_crit : float
            Critical distance.
        c_max_crit : float
            Constant for penalty minimum distance.
        penalty_max: array
            Array containing the penalty to add.
        """
        penalty_max = []
        for i in self.test_features:
            d_max = np.min(
                distance.cdist([i], self.train_features, 'euclidean'))
            p = 0.0
            if d_max > d_max_crit:
                p = c_max_crit * (d_max - d_max_crit)**2
            penalty_max.append(p)

        return penalty_max
