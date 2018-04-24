"""Greedy feature selection routines."""
from __future__ import print_function
from __future__ import absolute_import

import copy
import warnings
import multiprocessing
import time
from tqdm import trange, tqdm
import numpy as np
import json

from catlearn.cross_validation import k_fold

# Ignore warnings from cleaner progress tracking.
warnings.filterwarnings("ignore")


class GreedyElimination(object):
    """The greedy feature elimination class."""

    def __init__(self, nprocs=1, verbose=True, save_file=None):
        """Initialize the class.

        Parameters
        ----------
        nprocs : int
            Number of processers used in parallel implementation. Default is 1
            e.g. serial.
        verbose : bool
            Display some additional metrics on progress when set to True.
        """
        self.nprocs = nprocs
        self.verbose = verbose
        self.save_file = save_file

    def greedy_elimination(self, predict, features, targets, nsplit=2, step=1):
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
        features, targets, output, survivors, total_features = self._load_data(
            features, targets, nsplit)

        if self.verbose:
            # The tqdm package is used for tracking progress.
            iterator1 = trange(
                (total_features - 1) // step, desc='features eliminated ',
                leave=False)
        else:
            iterator1 = range((total_features - 1) // step)

        for fnum in iterator1:
            self.result = np.zeros((nsplit, total_features))
            meta = []

            if self.verbose:
                iterator2 = trange(nsplit, desc='k-folds             ',
                                   leave=False)
            else:
                iterator2 = range(nsplit)

            for self.index in iterator2:
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
                    meta_k = self._parallel_iterator(
                        d, train_features, test_features, train_targets,
                        test_targets, predict, meta_k)

                else:
                    meta_k = self._serial_iterator(
                        d, train_features, test_features, train_targets,
                        test_targets, predict, meta_k)

                if len(meta_k) > 0:
                    meta.append(meta_k)

            # Scores summed over k.
            scores = np.mean(self.result, axis=0)
            # Sort features according to score.
            s = np.argsort(scores)
            for g in range(step):
                eliminated = np.array([np.array(survivors)[s][g],
                                       np.array(scores)[s][g]]).tolist()
                if len(meta) > 0:
                    mean_meta = np.mean(meta, axis=0)
                    output.append(
                        np.concatenate([eliminated, float(mean_meta[g])],
                                       axis=0).tolist()
                    )
                else:
                    output.append(eliminated)
            # Delete features that, while missing gave the smallest error.
            survivors = [x for i, x in enumerate(survivors) if
                         i not in s[:step]]
            total_features -= step

            if self.save_file is not None:
                self._write_data(
                    output, survivors, total_features, features, targets)

        return output

    def _load_data(self, features, targets, nsplit):
        """Function to load or initialize data.

        Parameters
        ----------
        features : array
            The feature set for the training data.
        targets : array
            The targets for the traning data.
        nsplit : int
            The number of k-folds for the CV.

        Returns
        -------
        features : list
            List of k-fold feature arrays.
        targets : list
            List of k-fold target arrays.
        output : list
            The current list of output data.
        survivors : list
            The current list of surviving features.
        total_features : int
            The current number of surviving features.
        """
        # Make some k-fold splits.
        total_features = np.shape(features)[1]

        output = []
        survivors = list(range(total_features))
        load_data = False
        if self.save_file is not None:
            try:
                with open(self.save_file) as save_data:
                    data = json.load(save_data)
                    output = data['output']
                    survivors = data['survivors']
                    total_features = data['total_features']
                    features = [np.array(f) for f in data['features']]
                    targets = [np.array(t) for t in data['targets']]
                print('Resuming greedy search with {} features.'.format(
                    total_features))
                load_data = True
            except FileNotFoundError:
                print('Starting new greedy search.')

        if not load_data:
            features, targets = k_fold(
                features, targets=targets, nsplit=nsplit)

        return features, targets, output, survivors, total_features

    def _write_data(self, output, survivors, total_features, features,
                    targets):
        """Function to write data.

        Parameters
        ----------
        output : list
            The current list of output data.
        survivors : list
            The current list of surviving features.
        total_features : int
            The current number of surviving features.
        features : list
            List of k-fold feature arrays.
        targets : list
            List of k-fold target arrays.
        """
        data = {
            'output': output,
            'survivors': survivors,
            'total_features': total_features,
            'features': [f.tolist() for f in features],
            'targets': [t.tolist() for t in targets]
        }
        with open(self.save_file, 'w') as save_data:
            json.dump(data, save_data)

    def _parallel_iterator(self, d, train_features, test_features,
                           train_targets, test_targets, predict, meta_k):
        """Parallel iterator for the predictions.

        Parameters
        ----------
        d : int
            Dimension of the feature set.
        train_features : array
            The feature set for the training data.
        test_features : array
            The feature set for the test data.
        train_targets : array
            The training target data.
        test_targets : array
            The test target data:
        predict : object
            The prediction function.
        meta_k : list
            The metadata for the k-fold.

        Attributes
        ----------
        result : array
            The array holding the error associated with each subset of
            features.

        Returns
        -------
        meta_k : list
            The metadata for the k-fold.
        """
        pool = multiprocessing.Pool(self.nprocs)
        args = (
            (x, train_features, test_features, train_targets,
             test_targets, predict) for x in np.arange(d))
        for r in tqdm(pool.imap_unordered(
                _single_elimination, args), total=d,
                desc='nested              ', leave=False):
            self.result[self.index][r[0]] = r[1]
            if len(r) > 2:
                meta_k.append(r[2])
            # Wait to make things more stable.
            time.sleep(0.001)
        pool.close()

        return meta_k

    def _serial_iterator(self, d, train_features, test_features,
                         train_targets, test_targets, predict, meta_k):
        """Serial iterator for the predictions.

        Parameters
        ----------
        d : int
            Dimension of the feature set.
        train_features : array
            The feature set for the training data.
        test_features : array
            The feature set for the test data.
        train_targets : array
            The training target data.
        test_targets : array
            The test target data:
        predict : object
            The prediction function.
        meta_k : list
            The metadata for the k-fold.

        Attributes
        ----------
        result : array
            The array holding the error associated with each subset of
            features.

        Returns
        -------
        meta_k : list
            The metadata for the k-fold.
        """
        for x in trange(
                d, desc='nested              ', leave=False):
            args = (x, train_features, test_features,
                    train_targets, test_targets, predict)
            r = _single_elimination(args)
            self.result[self.index][r[0]] = r[1]
            if len(r) > 2:
                meta_k.append(r[2])

        return meta_k


def _single_elimination(args):
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
