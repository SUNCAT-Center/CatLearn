# -*- coding: utf-8 -*-
"""
AtoML 0.4.1
"""
from __future__ import print_function
import numpy as np


class SurrogateModel(object):
    """Base class for feature generation."""

    def __init__(self, train_model, predict, train_data, target):
        """Initialize the class.

        Parameters
        ----------
        train_model : object
            function which returns a trained regression model. This function
            should either train or update the regression model.
            Parameters
            ----------

            train_fp : array
                training data matrix.
            target : list
                training target feature.
        predict : object
            function which returns predictions, error estimates and meta data.
            Parameters
            ----------

            model : object
                train_model
            test_fp : array
                test data matrix.
            test_target : list
                test target feature.
        train_data : array
            training data matrix.
        target : list
            training target feature.
        """

        self.train_model = train_model
        self.predict = predict
        self.train_data = train_data
        self.target = target

    def test_acquisition(self, acquisition_function, initial_subset=None,
                         batch_size=1, objective='max'):
        """Return an array of test results for a surrogate model.
        """
        if initial_subset is None:
            train_index = list(range(2))
        else:
            train_index = initial_subset
        output = []
        for i in np.arange(len(self.target) // batch_size):
            # Setup data.
            test_index = np.delete(np.arange(len(self.train_data)),
                                   train_index)
            train_fp = self.train_data[train_index, :]
            train_target = np.array(self.target)[train_index]
            test_fp = self.train_data[test_index, :]
            test_target = np.array(self.target)[test_index]

            # Select a fitness reference.
            if objective == 'max':
                y_best = max(train_target)
            elif objective == 'min':
                y_best = min(train_target)
            elif isinstance(objective, float):
                y_best = objective

            if len(test_target) == 0:
                break
            # Do regression.
            model = self.train_model(train_fp, train_target)
            # Make predictions.
            score = self.predict(model, test_fp, test_target)
            y = score['prediction']
            std = score['uncertainty']
            # Calculate acquisition values.
            af = acquisition_function(y_best, y, std)
            sample = np.argsort(af)[::-1]

            # Append best candidates to be acquired.
            train_index += list(test_index[sample[:batch_size]])
            # Return meta data.
            output.append(score)
        return output

    def acquire(self, acquisition_function, unlabeled_data, aq_targets,
                initial_subset=None, batch_size=1):
        """Return indices of datapoints to acquire, from a known search space.
        """
        # Do regression.
        model = self.train_model(self.train_data, self.target)
        # Make predictions.
        score = self.predict(model, unlabeled_data)
        y = score['prediction']
        std = score['uncertainty']
        # Calculate acquisition values.
        af = acquisition_function(aq_targets, y, std)
        sample = np.argsort(af)[::-1]
        # Return best candidates and meta data.
        return list(sample[:batch_size]), score
