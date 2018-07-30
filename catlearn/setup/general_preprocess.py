"""A default setup for data preprocessing."""
import numpy as np

from catlearn.preprocess.clean_data import clean_variance, clean_infinite
from catlearn.preprocess.scaling import standardize


class GeneralPrepreprocess(object):
    """A general purpose data preprocessing class."""

    def __init__(self, clean_type='eliminate'):
        """Initialize the class.

        Parameters
        ----------
        clean_type : str
            Define method for handling missing data. Currently only elimination
            implemented.
        """
        self.clean_type = clean_type

    def process(self, train_features, train_targets, test_features=None):
        """Processing function.

        Parameters
        ----------
        train_features : array
            The array of training features.
        train_targets : array
            A list of training target values.
        test_features : array
            The array of test features.
        """
        if self.clean_type is 'eliminate':
            train_features, train_targets, test_features = \
                self._eliminate_cleaner(
                    train_features, train_targets, test_features)
            if len(train_features) == 0:
                raise AssertionError("All features has been eliminated.")
        else:
            raise NotImplementedError

        train_features, test_features = self._standardize_scalar(
            train_features, test_features)

        return train_features, train_targets, test_features

    def transform(self, features):
        """Function to transform a new set of features.

        Parameters
        ----------
        features : array
            A new array of features to clean. This will most likely be the new
            test features.

        Returns
        -------
        processed : array
            A cleaned and scaled feature set.
        """
        features = np.array(features)

        # Check to make sure the required data is attached to class.
        if not hasattr(self, 'scale_mean'):
            msg = 'Must run the process function first.'
            raise AttributeError(msg)

        # First eliminate features.
        if self.clean_type is 'eliminate':
            processed = features[:, self.clean_index]

        # Scale the features.
        processed = (processed - self.scale_mean) / self.scale_std

        return processed

    def _eliminate_cleaner(self, train_features, train_targets, test_features):
        """Function to remove data missing or useless.

        Parameters
        ----------
        train_features : array
            The array of training features.
        train_targets : array
            A list of training target values.
        test_features : array
            The array of test features.
        """
        train_features = np.array(train_features)
        if test_features is not None:
            test_features = np.array(test_features)

        # Identify clean and informative features.
        finite = clean_infinite(train=train_features,
                                test=test_features,
                                targets=train_targets)
        informative = clean_variance(train=train_features,
                                     test=test_features)

        # Join lists of features to keep.
        self.clean_index = np.intersect1d(finite['index'],
                                          informative['index'])

        if test_features is None:
            return train_features[:, self.clean_index], train_targets, \
                test_features

        return train_features[:, self.clean_index], train_targets, \
            test_features[:, self.clean_index]

    def _standardize_scalar(self, train_features, test_features):
        """Function to feature data.

        Parameters
        ----------
        train_features : array
            The array of training features.
        test_features : array
            The array of test features.
        """
        std = standardize(train_features, test_features)

        self.scale_mean = std['mean']
        self.scale_std = std['std']

        return std['train'], std['test']
