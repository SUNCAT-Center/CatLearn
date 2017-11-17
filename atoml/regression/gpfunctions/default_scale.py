"""Class to scale everything within regression functions."""
from atoml.preprocess.feature_preprocess import standardize
from atoml.preprocess.scale_target import target_standardize

class ScaleData(object):
    def __init__(self, train_features, train_targets):
        self.train_features = train_features
        self.train_targets = train_targets

    def train(self):
        """Scale the training features and targets."""
        self.feature_data = standardize(
            train_matrix=self.train_features, mean=None, std=None, local=True
            )

        self.target_data = target_standardize(target=self.train_targets)

        return self.feature_data['train'], self.target_data['target']

    def test(self, test_features):
        """Scale the test features.

        Parameters
        ----------
        test_features : array
            Feature matrix for the test data.
        """
        scaled = (test_features - self.feature_data['mean']) \
            * self.feature_data['std']

        return scaled

    def rescale(self, predictions):
        """Rescale predictions.

        Parameters
        ----------
        predictions : list
           The predicted values from the GP.
        """
        p = predictions * self.target_data['std'] + self.target_data['mean']

        return p
