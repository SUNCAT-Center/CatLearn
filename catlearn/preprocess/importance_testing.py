"""Functions to check feature significance."""
from __future__ import absolute_import
from __future__ import division

import copy
import multiprocessing
import time
from tqdm import trange, tqdm
import numpy as np

from catlearn.cross_validation import k_fold


class ImportanceElimination(object):
    """The feature importance elimination class."""

    def __init__(self, transform, nprocs=1, verbose=True):
        """Initialize the class.

        Parameters
        ----------
        nprocs : int
            Number of processers used in parallel implementation. Default is 1
            e.g. serial.
        verbose : bool
            Display some additional metrics on progress when set to True.
        """
        self.transform = transform
        self.nprocs = nprocs
        self.verbose = verbose

    def importance_elimination(self, train_predict, test_predict, features,
                               targets, nsplit=2, step=1):
        """Importance feature elimination.

        Function to iterate through feature set, eliminating least important
        feature in each pass. This is the backwards elimination algorithm.

        Parameters
        ----------
        train_predict : object
            A function that will train a model. The function should accept
            the parameters:

                train_features : array
                train_targets : list

            predict should return a function that can be passed to
            test_predict.
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

                pred = train_predict(train_features, train_targets)

                _, d = np.shape(train_features)
                meta_k = []

                # Iterate through features and find error for removing it.
                if self.nprocs != 1:
                    meta_k = self._parallel_iterator(
                        d, train_features, test_features, train_targets,
                        test_targets, pred, test_predict, meta_k)

                else:
                    meta_k = self._serial_iterator(
                        d, train_features, test_features, train_targets,
                        test_targets, pred, test_predict, meta_k)

                if len(meta_k) > 0:
                    meta.append(meta_k)

            # Scores summed over k.
            scores = np.mean(self.result, axis=0)
            # Sort features according to score.
            s = np.argsort(scores)
            for g in range(step):
                eliminated = [np.array(survivors)[s][g],
                              np.array(scores)[s][g]]
                if len(meta) > 0:
                    mean_meta = np.mean(meta, axis=0)
                    output.append(np.concatenate([eliminated, mean_meta[g]],
                                                 axis=0))
                else:
                    output.append(eliminated)
            # Delete features that, while missing gave the smallest error.
            survivors = [x for i, x in enumerate(survivors) if
                         i not in s[:step]]
            total_features -= step

        return output

    def _parallel_iterator(self, d, train_features, test_features,
                           train_targets, test_targets, predict, test_predict,
                           meta_k):
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
             test_targets, predict, self.transform, test_predict
             ) for x in np.arange(d))
        for r in tqdm(pool.imap_unordered(
                _predictor, args), total=d,
                desc='nested              ', leave=False):
            self.result[self.index][r[0]] = r[1]
            if len(r) > 2:
                meta_k.append(r[2])
            # Wait to make things more stable.
            time.sleep(0.001)
        pool.close()

        return meta_k

    def _serial_iterator(self, d, train_features, test_features,
                         train_targets, test_targets, predict, test_predict,
                         meta_k):
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
                    train_targets, test_targets, predict, self.transform,
                    test_predict)
            r = _predictor(args)
            self.result[self.index][r[0]] = r[1]
            if len(r) > 2:
                meta_k.append(r[2])

        return meta_k


def _predictor(args):
    """Make a feature random noise.

    Parameters
    ----------
    features : array
        The original feature matrix.
    index : int
        The index of the feature to be transformed.

    Returns
    -------
    features : array
        Feature matrix with an invariant feature column in matrix.
    """
    # Unpack args tuple.
    f = args[0]
    train_features = args[1]
    test_features = args[2]
    train_targets = args[3]
    test_targets = args[4]
    predict = args[5]
    transform = args[6]
    test_predict = args[7]

    train, test = transform((f, train_features, test_features))

    # Calculate the error on predictions.
    result = test_predict(predict, test, test_targets)

    if isinstance(result, list):
        error = result[0]
        meta = result[1:]
        return f, error, meta

    return f, result


def feature_invariance(args):
    """Make a feature invariant.

    Parameters
    ----------
    features : array
        The original feature matrix.
    index : int
        The index of the feature to be transformed.

    Returns
    -------
    features : array
        Feature matrix with an invariant feature column in matrix.
    """
    # Unpack args tuple.
    f = args[0]
    train_features = args[1]
    test_features = args[2]

    # Transform required index.
    all_features = np.concatenate((train_features, test_features), axis=0)
    m = np.mean(all_features[:, f])
    s = np.std(all_features[:, f])

    train = train_features.copy()
    train[:, f] = s + m

    test = test_features.copy()
    test[:, f] = s + m

    return train, test


def feature_randomize(args):
    """Make a feature random noise.

    Parameters
    ----------
    features : array
        The original feature matrix.
    index : int
        The index of the feature to be transformed.

    Returns
    -------
    features : array
        Feature matrix with an invariant feature column in matrix.
    """
    # Unpack args tuple.
    f = args[0]
    train_features = args[1]
    test_features = args[2]

    # Transform required index.
    all_features = np.concatenate((train_features, test_features), axis=0)
    m = np.mean(all_features[:, f])
    s = np.std(all_features[:, f])

    train = train_features.copy()
    r = np.random.random(train.shape[0])
    train[:, f] = (r * s) + m

    test = test_features.copy()
    r = np.random.random(test.shape[0])
    test[:, f] = (r * s) + m

    return train, test


def feature_shuffle(args):
    """Shuffle a feature.

    The method has a number of advantages for measuring feature importance.
    Notably the original values and scale of the feature are maintained.

    Parameters
    ----------
    features : array
        The original feature matrix.
    index : int
        The index of the feature to be shuffled.

    Returns
    -------
    features : array
        Feature matrix with a shuffled feature column in matrix.
    """
    # Unpack args tuple.
    f = args[0]
    train_features = args[1]
    test_features = args[2]

    # Transform required index.
    train = train_features.copy()
    np.random.shuffle(train[:, f])

    test = test_features.copy()
    np.random.shuffle(test[:, f])

    return train, test
