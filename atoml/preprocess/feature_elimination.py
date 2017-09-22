"""Functions to select features for the fingerprint vectors."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import defaultdict
from math import log

from atoml.regression.scikit_wrapper import RegressionFit


class FeatureScreening(object):
    """Class for feature elimination based on correlation screening."""

    def __init__(self, correlation='pearson', iterative=True,
                 regression='ridge'):
        """Screening setup.

        Parameters
        ----------
        correlation : str
            Type of correlation to use. Can be pearson, spearman or kendall.
            Default is pearson.
        iterative : boolean
            Define whether to perform the iterative version of the screening
            routine. Default is True.
        regression : str
            Define regression method for ordering features. Can be ridge,
            elastic or lasso. Default is ridge regression.
        """
        self.correlation = correlation
        self.iterative = iterative
        self.regression = regression

    def screen(self, target, feature_matrix):
        """Feature selection based on SIS.

        Further discussion on this topic can be found in Fan, J., Lv, J., J. R.
        Stat. Soc.: Series B, 2008, 70, 849.

        Parameters
        ----------
        target : list
            The target values for the training data.
        feature_matrix : array
            The feature matrix for the training data.

        Returns
        -------
        index : list
            The ordered list of feature indices.
        correlation : list
            The ordered list of correlations between features and targets.
        """
        select = defaultdict(list)

        # Get correlation between each feature and the targets.
        corr, order = self._get_correlation(target=target,
                                            feature_matrix=feature_matrix)

        # Order everything highest correlated to least.
        sort_list = [list(i) for i in zip(*sorted(zip(np.abs(corr), order),
                                                  key=lambda x: x[0],
                                                  reverse=True))]

        select['index'] = sort_list[1]
        select['correlation'] = sort_list[0]

        return select

    def iterative_screen(self, target, feature_matrix, size=None, step=None):
        """Function iteratively screen featues.

        Parameters
        ----------
        target : list
            The target values for the training data.
        feature_matrix : array
            The feature matrix for the training data.
        size : int
            Number of features to be returned. Default is number of data.
        step : int
            Step size by which to reduce the number of features. Default is
            n / log(n).

        Returns
        -------
        index : list
            The ordered list of feature indices, top index[:size] will be
            indices for best features.
        """
        n, f = np.shape(feature_matrix)
        select = defaultdict(list)
        accepted, rejected = [], list(range(f))
        keep_train = feature_matrix

        # Assign default values and/or perform sanity checks.
        size, step = self._dimension_check(n=n, f=f, size=size, step=step)

        # Initiate the feature reduction.
        iscreen = self.screen(target=target, feature_matrix=feature_matrix)
        rscreen = self._reduce_matrix(feature_matrix=feature_matrix,
                                      index_order=iscreen['index'], size=size)

        # Get ordering of remaining features from linear regression.
        regr = self._regression_ordering(target=target,
                                         feature_matrix=rscreen['matrix'],
                                         size=step, steps=500)

        # Sort all new ordering of accepted and rejected features.
        sis_accept = [rscreen['accepted'][i] for i in regr]
        for i in sorted(sis_accept, reverse=True):
            accepted.append(rejected[i])
            del rejected[i]

        # Create new active feature matrix.
        feature_train = np.delete(keep_train, accepted, axis=1)
        feature_reduced = np.delete(keep_train, rejected, axis=1)

        # Iteratively reduce the remaining number of features by the step size.
        while np.shape(feature_reduced)[1] < size:
            # Calculate step/size remainder if necessary.
            if size - np.shape(feature_reduced)[1] < step:
                step = size - np.shape(feature_reduced)[1]
            # Calculate the residual for the remaining features.
            response = self._get_response(train_matrix=feature_train,
                                          reduced_matrix=feature_reduced)

            # Do screening analysis on the residuals.
            iscreen = self.screen(target=target, feature_matrix=response)
            rscreen = self._reduce_matrix(feature_matrix=response,
                                          index_order=iscreen['index'],
                                          size=size)

            # Get ordering of remaining features from linear regression.
            regr = self._regression_ordering(target=target,
                                             feature_matrix=rscreen['matrix'],
                                             size=step, steps=500)

            # Sort all new ordering of accepted and rejected features.
            sis_accept = [rscreen['accepted'][i] for i in regr]
            for i in sorted(sis_accept, reverse=True):
                accepted.append(rejected[i])
                del rejected[i]

            # Create new active feature matrix.
            feature_train = np.delete(keep_train, accepted, axis=1)
            feature_reduced = np.delete(keep_train, rejected, axis=1)

        select['index'] = list(accepted) + list(rejected)

        return select

    def eliminate_features(self, target, train_features, test_features,
                           size=None, step=None, order=None):
        """Function to eliminate features from training/test data.

        Parameters
        ----------
        target : list
            The target values for the training data.
        train_features : array
            Array of training data to eliminate features from.
        test_features : array
            Array of test data to eliminate features from.
        size : int
            Number of features after elimination.
        step : int
            Number of features to eliminate at each step.
        order : list
            Precomputed ordered indices for features.

        Returns
        -------
        reduced_train : array
            Reduced training feature matrix, now n x size shape.
        reduced_test : array
            Reduced test feature matrix, now m x size shape.
        """
        if order is None:
            if self.iterative:
                order = self.iterative_screen(target=target,
                                              feature_matrix=train_features,
                                              size=size, step=step)['index']
            else:
                order = self.screen(target=target,
                                    feature_matrix=train_features)['index']

        reduced_train = self._reduce_matrix(feature_matrix=train_features,
                                            index_order=order, size=size)
        reduced_test = self._reduce_matrix(feature_matrix=test_features,
                                           index_order=order, size=size)

        return reduced_train['matrix'], reduced_test['matrix']

    def _get_correlation(self, target, feature_matrix):
        """Function to return the correlation between features and targets.

        Parameters
        ----------
        target : list
            The target values for the training data.
        feature_matrix : array
            Array to eliminate features from.
        """
        # Probably easier to reshape target than take transpose of feature
        # matrix, should be tested.
        t = np.reshape(target, (len(target), 1))

        corr = []
        order = list(range(np.shape(feature_matrix)[1]))
        for i in order:
            d = feature_matrix[:, i:i+1]
            if np.allclose(d, d[0]):
                corr.append(0.)
            elif self.correlation is 'pearson':
                corr.append(pearsonr(x=d, y=t)[0])
            elif self.correlation is 'spearman':
                corr.append(spearmanr(a=d, b=t)[0])
            elif self.correlation is 'kendall':
                corr.append(kendalltau(x=d, y=t)[0])

        return corr, order

    def _dimension_check(self, n, f, size, step):
        """Some sanity checks before doing iterative feature elimination.

        Parameters
        ----------
        n : int
            Number of data points.
        f : int
            Number of features.
        size : int
            Number of features after elimination.
        step : int
            Number of features to eliminate at each step.
        """
        if size is None:
            size = n

        if self.iterative:
            msg = 'Not enough features to perform screening requires 2 x size.'
            assert f > 2 * size, msg

            if step is None:
                step = round(n / log(n))
            msg = 'Step size is too large, reduce it.'
            assert step < size, msg

        else:
            msg = 'Not enough features to perform screening.'
            assert f > size, msg

        return size, step

    def _reduce_matrix(self, feature_matrix, index_order, size):
        """Function to eliminate features from matrix.

        Parameters
        ----------
        feature_matrix : array
            Array to eliminate features from.
        index_order : list
            Feature index ordered according best to worst.
        size : int
            Number of features to remain in the matrix.
        """
        data = defaultdict(list)
        data['accepted'] = index_order[:size]
        data['rejected'] = index_order[size:]
        data['matrix'] = np.delete(feature_matrix, data['rejected'], axis=1)

        return data

    def _regression_ordering(self, target, feature_matrix, size, steps):
        """Function to get ordering of features absed on linear regression."""
        # Set up the regression fitting function.
        rf = RegressionFit(train_matrix=feature_matrix, train_target=target,
                           method=self.regression)

        order = rf.feature_select(size=size, iterations=1e5, steps=steps,
                                  line_search=True, min_alpha=1.e-8,
                                  max_alpha=1.e-1, eps=1e-3)

        return order['accepted']

    def _get_response(self, train_matrix, reduced_matrix):
        """Function to calculate the uncorrelated response of features.

        Parameters
        ----------
        feature_matrix : array
            Array to eliminate features from.
        """
        response = []
        nv = np.linalg.norm(np.transpose(reduced_matrix), axis=1)
        for d in np.transpose(train_matrix):
            for a, n in zip(np.transpose(reduced_matrix), nv):
                r = (d - np.dot(a, np.dot(d, a))) / (n ** 2)
            response.append(r)

        return np.transpose(response)
