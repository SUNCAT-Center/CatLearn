"""Greedy feature selection routines."""
from __future__ import print_function
from __future__ import absolute_import

import copy
import warnings
import multiprocessing
from tqdm import tqdm
import numpy as np

from atoml.cross_validation import k_fold

# Ignore warnings from cleaner progress tracking.
warnings.filterwarnings("ignore")


class GreedyElimination(object):
    def __init__(self, direction='backward'):
        self.direction = direction

    def greedy_elimination(self, predict, features, targets, nsplit=2):
        """Greedy feature elimination.

        Function to iterate through feature set, eliminating worst feature in
        each pass.

        Parameters
        ----------
        predict : object
            A function that will make the predictions. This should expect to be
            passed training and testing features and targets.
        features : array
            An n, d array of features.
        targets : list
            A list of the target features.
        nsplit : int
            Number of folds in k-fold cross-validation.

        Returns
        -------
        size_result : dict
            The dictionary contains the averaged error over the specified
            k-fold data sets.
        """
        features, targets = k_fold(features, targets, nsplit)
        _, total_features = np.shape(features[0])

        size_result = {}

        print('starting greedy feature elimination')
        # The tqdm package is used for tracking progress.
        for fnum in tqdm(range(total_features - 1),
                         desc='features eliminated'):
            self.result = np.zeros((nsplit, total_features))
            for self.index in range(nsplit):
                # Sort out training and testing data.
                train_features = copy.deepcopy(features)
                train_targets = copy.deepcopy(targets)
                test_features = train_features.pop(self.index)
                test_targets = train_targets.pop(self.index)

                train_features = np.concatenate(train_features, axis=0)
                train_targets = np.concatenate(train_targets, axis=0)

                _, d = np.shape(train_features)

                # Iterate through features and find error for removing it.
                pool = multiprocessing.Pool(None)
                tasks = np.arange(d)
                args = (
                    (x, train_features, test_features, train_targets,
                     test_targets, predict) for x in tasks)
                parallel_iterate = pool.map_async(
                    self._single_elimination, args,
                    callback=self._elim_callback)
                parallel_iterate.wait()
                # print(self.result)
                # for f in range(d):
                #    result[index][f] = _single_elimination(
                #        f, train_features, test_features, train_targets,
                #        test_targets, predict)

            # Delete feature that gives largest error.
            for self.index in range(nsplit):
                features[self.index] = np.delete(
                    features[self.index], np.argmin(
                        np.sum(self.result, axis=0)), axis=1)
            total_features -= 1

            size_result[d] = np.sum(self.result) / (d * nsplit)

        return size_result

    def _single_elimination(self, args):
        """Function to eliminate a single feature and make a prediction."""
        f = args[0]
        train_features = args[1]
        test_features = args[2]
        train_targets = args[3]
        test_targets = args[4]
        predict = args[5]

        train = np.delete(train_features, f, axis=1)
        test = np.delete(test_features, f, axis=1)

        error = predict(train, train_targets, test, test_targets)

        return f, error

    def _elim_callback(self, x):
        for i in x:
            self.result[self.index][i[0]] = i[1]
