"""Greedy feature selection routines."""
from __future__ import print_function
from __future__ import absolute_import

import copy
import warnings
from tqdm import tqdm
import numpy as np

from atoml.cross_validation import k_fold

# Ignore warnings from cleaner progress tracking.
warnings.filterwarnings("ignore")


def greedy_elimination(predict, features, targets, nsplit=2):
    """Greedy feature elimination.

    Function to iterate through feature set, eliminating worst feature in each
    pass.

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
        The dictionary contains the averaged error over the specified k-fold
        data sets.
    """
    features, targets = k_fold(features, targets, nsplit)
    _, total_features = np.shape(features[0])

    size_result = {}

    print('starting greedy feature elimination')
    # The tqdm package is used for tracking progress.
    for fnum in tqdm(range(total_features - 1), desc='features eliminated'):
        result = np.zeros((nsplit, total_features))
        for index in range(nsplit):
            # Sort out training and testing data.
            train_features = copy.deepcopy(features)
            train_targets = copy.deepcopy(targets)
            test_features = train_features.pop(index)
            test_targets = train_targets.pop(index)

            train_features = np.concatenate(train_features, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)

            _, d = np.shape(train_features)

            # Iterate through features and find error for removing it.
            for f in range(d):
                train = np.delete(train_features, f, axis=1)
                test = np.delete(test_features, f, axis=1)

                result[index][f] = predict(
                    train, train_targets, test, test_targets)

        # Delete feature that gives largest error.
        for index in range(nsplit):
            features[index] = np.delete(
                features[index], np.argmin(np.sum(result, axis=0)), axis=1)
        total_features -= 1

        size_result[d] = np.sum(result) / (d * nsplit)

    return size_result
