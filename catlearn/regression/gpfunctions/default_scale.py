"""Scale everything within regression functions."""
from __future__ import absolute_import
from __future__ import division

import numpy as np

from catlearn.preprocess.scaling import standardize, target_standardize


class ScaleData(object):
    """Class to perform default scaling in the regression functions.

    Will standardize both the features and the targets. These can then be
    rescaled before being returned. The parameters can be accessed from the
    class with:

        ScaleData.feature_data['mean']

    This can be accessed from the gp with:

        gp = GaussianProcess(...)
        gp.scaling.feature_data['mean']
    """

    def __init__(self, train_features, train_targets):
        """Initialize the scaling routine.

        Parameters
        ----------
        train_features : array
            Array of training features.
        train_targets : list
            List of the training targets.
        """
        self.train_features = np.asarray(train_features)
        self.train_targets = np.asarray(train_targets)

    def train(self):
        """Scale the training features and targets.

        Returns
        -------
        feature_data : array
            The scaled features for the training data.
        target_data : array
            The scaled targets for the training data.
        """
        self.feature_data = standardize(train_matrix=self.train_features)

        self.target_data = target_standardize(target=self.train_targets)

        return self.feature_data['train'], self.target_data['target']

    def test(self, test_features):
        """Scale the test features.

        Parameters
        ----------
        test_features : array
            Feature matrix for the test data.

        Returns
        -------
        scaled_features : array
            The scaled features for the test data.
        """
        test_features = np.asarray(test_features)
        center = test_features - self.feature_data['mean']
        scaled_features = center / self.feature_data['std']

        return scaled_features

    def rescale_targets(self, predictions):
        """Rescale predictions.

        Parameters
        ----------
        predictions : list
            The predicted values from the GP.

        Returns
        -------
        p : array
            The rescaled predictions.
        """
        predictions = np.asarray(predictions)
        p = (predictions * self.target_data['std']) + self.target_data['mean']

        return p
