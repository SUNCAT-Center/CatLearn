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
            Define method for handling missing data. Currntly only elimination
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
        # Check to make sure the required data is attached to class.
        if not hasattr(self, 'scale_mean'):
            msg = 'Must run the process function first.'
            raise AttributeError(msg)

        # First eliminate features.
        if self.clean_type is 'eliminate':
            processed = np.delete(features, self.eliminate_index, axis=1)

        # Then scale the features.
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
        reshape_targets = False
        if len(np.shape(train_targets)) == 1:
            train_targets = np.reshape(train_targets, (len(train_targets), 1))
            reshape_targets = True

        cleaned = clean_infinite(train=train_features, test=test_features,
                                 targets=train_targets)

        # Assign cleaned training target data.
        train_targets = cleaned['targets']
        if reshape_targets:
            train_targets = np.reshape(train_targets, (len(train_targets),))

        # Keep the indexes to delete.
        self.eliminate_index = cleaned['index']

        cleaned = clean_variance(train=cleaned['train'], test=cleaned['test'])

        # Join lists of features to eliminate.
        self.eliminate_index += cleaned['index']

        return cleaned['train'], train_targets, cleaned['test']

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
