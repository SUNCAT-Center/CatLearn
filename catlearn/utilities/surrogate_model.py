# -*- coding: utf-8 -*-
"""
AtoML 0.4.1
"""
from __future__ import print_function
import numpy as np
from tqdm import tqdm


class SurrogateModel(object):
    """Base class for feature generation."""

    def __init__(self, train_model, predict, acquisition_function,
                 train_data, target):
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

            Returns
            ----------
            model : object
                Trained model object, which can be accepted by predict.

        predict : object
            function which returns predictions, error estimates and meta data.

            Parameters
            ----------

            model : object
                model returned by train_model.
            test_fp : array
                test data matrix.
            test_target : list
                test target feature.

            Returns
            ----------
            acquition_args : list
                ordered list of arguments for aqcuisition_function.
            score : object
                arbitratry meta data for output.

        acquisition_function : object
            function which returns a list of acquisition function values,
            where the largest value(s) are to be acquired.

            Parameters
            ----------

            *aqcuisition_args

            Returns
            ----------
            af : list
                Acquisition function values corresponding.

        train_data : array
            training data matrix.
        target : list
            training target feature.
        """

        self.train_model = train_model
        self.predict = predict
        self.acquisition_function = acquisition_function
        self.train_data = train_data
        self.target = target

    def test_acquisition(self, initial_subset=None, batch_size=1, n_max=None):
        """Return an array of test results for a surrogate model.

        Parameters
        ----------
        initial_subset : list
            Row indices of data to train on in the first iteration.
        batch_size : int
            Number of training points to acquire (move from test to training)
            in every iteration.
        n_max : int
            Max number of training points to test.
        """
        if initial_subset is None:
            train_index = list(range(max(batch_size, 2)))
        else:
            train_index = initial_subset

        if n_max is None:
            n_max = len(self.target)

        output = []
        for i in tqdm(np.arange(n_max // batch_size)):
            # Setup data.
            test_index = np.delete(np.arange(len(self.train_data)),
                                   np.array(train_index))
            train_fp = np.array(self.train_data)[train_index, :]
            train_target = np.array(self.target)[train_index]
            test_fp = np.array(self.train_data)[test_index, :]
            test_target = np.array(self.target)[test_index]

            if len(test_target) == 0:
                break
            elif len(test_target) < batch_size:
                batch_size = len(test_target)
            # Do regression.
            model = self.train_model(train_fp, train_target)

            # Make predictions.
            aqcuisition_args, score = self.predict(model, test_fp, test_target)

            # Calculate acquisition values.
            af = self.acquisition_function(*aqcuisition_args)
            sample = np.argsort(af)[::-1]

            to_acquire = test_index[sample[:batch_size]]
            assert len(to_acquire) == batch_size

            # Append best candidates to be acquired.
            train_index += list(to_acquire)
            # Return meta data.
            output.append(score)
        return output

    def acquire(self, unlabeled_data, initial_subset=None, batch_size=1):
        """Return indices of datapoints to acquire, from a known search space.

        Parameters
        ----------
        unlabeled_data : array
            Data matrix representing an unlabeled search space.
        initial_subset : list
            Row indices of data to train on in the first iteration.
        batch_size : int
            Number of training points to acquire (move from test to training)
            in every iteration.
        """
        # Do regression.
        model = self.train_model(self.train_data, self.target)
        # Make predictions.
        aqcuisition_args, score = self.predict(model, unlabeled_data)

        # Calculate acquisition values.
        af = self.acquisition_function(*aqcuisition_args)
        sample = np.argsort(af)[::-1]

        to_acquire = sample[:batch_size]
        assert len(to_acquire) == batch_size

        # Return best candidates and meta data.
        return list(to_acquire), score
