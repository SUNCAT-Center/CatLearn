"""Greedy feature selection routines."""
from __future__ import print_function
from __future__ import absolute_import

import copy
import warnings
import multiprocessing
import time
from tqdm import trange, tqdm
import numpy as np

from atoml.cross_validation import k_fold

# Ignore warnings from cleaner progress tracking.
warnings.filterwarnings("ignore")


class GreedyElimination(object):
    """The greedy feature elimination class."""

    def __init__(self, nprocs=1):
        """Initialize the class.

        Parameters
        ----------
        nprocs : int
            Number of processers used in parallel implementation. Default is 1
            e.g. serial.
        """
        self.nprocs = nprocs

    def greedy_elimination(self, predict, features, targets, nsplit=2):
        """Greedy feature elimination.

        Function to iterate through feature set, eliminating worst feature in
        each pass. This is the backwards greedy algorithm.

        Parameters
        ----------
        predict : object
            A function that will make the predictions. This should expect to be
            passed training and testing features and targets.
        features : array
            An n, d array of features.
        targets : list
            A list of the target values.
        nsplit : int
            Number of folds in k-fold cross-validation.

        Returns
        -------
        size_result : dict
            The dictionary contains the averaged error over the specified
            k-fold data sets.
        """
        # Make some k-fold splits.
        features, targets = k_fold(features, targets=targets, nsplit=nsplit)
        _, total_features = np.shape(features[0])

        size_result = {}
        eliminated = []
        survivors = list(range(total_features))

        print('starting greedy feature elimination')
        # The tqdm package is used for tracking progress.
        for fnum in trange(total_features - 1, desc='features eliminated '):
            self.result = np.zeros((nsplit, total_features))
            for self.index in trange(nsplit, desc='k-folds             ',
                                     leave=False):
                # Sort out training and testing data.
                train_features = copy.deepcopy(features)
                train_targets = copy.deepcopy(targets)
                test_features = train_features.pop(self.index)[:, survivors]
                test_targets = train_targets.pop(self.index)

                train_features = np.concatenate(train_features,
                                                axis=0)[:, survivors]
                train_targets = np.concatenate(train_targets, axis=0)

                _, d = np.shape(train_features)

                # Iterate through features and find error for removing it.
                if self.nprocs != 1:
                    # First a parallel implementation.
                    pool = multiprocessing.Pool(self.nprocs)
                    tasks = np.arange(d)
                    args = (
                        (x, train_features, test_features, train_targets,
                         test_targets, predict) for x in tasks)
                    for r in tqdm(pool.imap_unordered(
                            self._single_elimination, args), total=d,
                            desc='nested              ', leave=False):
                        self.result[self.index][r[0]] = r[1]
                        # Wait to make things more stable.
                        time.sleep(0.001)
                    pool.close()

                else:
                    # Then a more clear serial implementation.
                    for x in trange(
                            nsplit, desc='nested              ', leave=False):
                        args = (x, train_features, test_features,
                                train_targets, test_targets, predict)
                        r = self._single_elimination(args)
                        self.result[self.index][r[0]] = r[1]

            # Scores summed over k.
            scores = np.sum(self.result, axis=0)
            # Delete feature that, while missing gave the smallest error.
            worst = np.argmin(scores)
            survivors.pop(int(worst))
            eliminated.append([worst, scores[worst]])
            total_features -= 1

            # Average the error over the k-fold data.
            size_result[d] = np.sum(self.result) / (d * nsplit)

        return eliminated

    def _single_elimination(self, args):
        """Function to eliminate a single feature and make a prediction.

        Parameters
        ----------
        args : tuple
            Parameters and data to be passed to elimination function.

        Returns
        -------
        f : int
            Feature index being eliminated.
        error : float
            The averaged error for the test data.
        """
        # Unpack args tuple.
        f = args[0]
        train_features = args[1]
        test_features = args[2]
        train_targets = args[3]
        test_targets = args[4]
        predict = args[5]

        # Delete required index.
        train = np.delete(train_features, f, axis=1)
        test = np.delete(test_features, f, axis=1)

        # Calculate the error on predictions.
        error = predict(train, train_targets, test, test_targets)

        return f, error
