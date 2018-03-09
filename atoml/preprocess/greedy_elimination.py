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
            A function that will make the predictions. predict should accept
            the parameters:
                train_features : array
                test_features : array
                train_targets : list
                test_targets : list
            predict should return either a float or a list of floats. The float
            or the first value of the list will be used as the fitness score.
        features : array
            An n, d array of features.
        targets : list
            A list of the target values.
        nsplit : int
            Number of folds in k-fold cross-validation.

        Returns
        -------
        output : array
            First column is the index of features in the order they were
            eliminated.
            Second column are corresponding cost function values, averaged over
            the k fold split.
            Following columns are any additional values returned by predict,
            averaged over the k fold split.
        """
        # Make some k-fold splits.
        features, targets = k_fold(features, targets=targets, nsplit=nsplit)
        _, total_features = np.shape(features[0])

        output = []
        survivors = list(range(total_features))

        print('starting greedy feature elimination')
        # The tqdm package is used for tracking progress.
        for fnum in trange(total_features - 1, desc='features eliminated '):
            self.result = np.zeros((nsplit, total_features))
            meta = []
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
                meta_k = []

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
                        if len(r) > 2:
                            meta_k.append(r[2])
                        # Wait to make things more stable.
                        time.sleep(0.001)
                    pool.close()

                else:
                    # Then a more clear serial implementation.
                    for x in trange(
                            d, desc='nested              ', leave=False):
                        args = (x, train_features, test_features,
                                train_targets, test_targets, predict)
                        r = self._single_elimination(args)
                        self.result[self.index][r[0]] = r[1]
                        if len(r) > 2:
                            meta_k.append(r[2])
                if len(meta_k) > 0:
                    meta.append(meta_k)

            # Scores summed over k.
            scores = np.mean(self.result, axis=0)
            # Delete feature that, while missing gave the smallest error.
            i = np.argmin(scores)
            worst = survivors.pop(int(i))
            eliminated = [worst, scores[i]]
            if len(meta) > 0:
                mean_meta = np.mean(meta, axis=0)
                output.append(np.concatenate([eliminated, mean_meta[i]],
                                             axis=0))
            else:
                output.append(eliminated)
            total_features -= 1

        return output

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
            A cost function.
            Typically the log marginal likelihood or goodness of fit.
        meta : list
            Additional optional values. Typically cross validation scores.
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
        result = predict(train, train_targets, test, test_targets)

        if isinstance(result, list):
            error = result[0]
            meta = result[1:]
            return f, error, meta
        return f, result
