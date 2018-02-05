"""GP acquisition functions."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.special import erf
from collections import defaultdict, Counter

from atoml.utilities.clustering import cluster_features


class AcquisitionFunctions(object):
    """Base class for acquisition functions."""

    def __init__(self, targets, predictions, uncertainty, train_features=None,
                 test_features=None, k=3):
        """Initialization of class.

        Parameters
        ----------
        targets : list
            List of known target values.
        predictions : list
            List of predictions from the GP.
        uncertainty : list
            List of variance on the GP predictions.
        train_features : array
            Feature matrix for the training data.
        test_features : array
            Feature matrix for the test data.
        k : int
            Number of cluster to generate with clustering.
        """
        self.targets = targets
        self.predictions = predictions
        self.uncertainty = uncertainty
        self.train_features = train_features
        self.test_features = test_features
        self.k = k

    def rank(self, x='max', metrics=['cdf', 'optimistic']):
        """Rank predictions based on acquisition function.

        Returns
        -------
        res : dict
            A dictionary of lists containg the fitness of each test point for
            the different acquisition functions.
        metrics : list
            list of strings.
            Accepted values are 'cdf', 'optimistic' and 'gaussian'.
        """
        # Create dictionary with a list for each acquisition function.
        res = {}
        for key in metrics:
            res.update({key: []})
        # Select a fitness reference.
        if x == 'max':
            best = max(self.targets)
        elif x == 'min':
            best = min(self.targets)
        elif isinstance(x, float):
            best = x

        # Calcuate fitness based on acquisition functions.
        for i, j in zip(self.predictions, self.uncertainty):
            res['cdf'].append(self._cdf_fit(x=best, m=i, v=j))
            res['optimistic'].append(self._optimistic_fit(x=best, m=i, v=j))
            if 'gaussian' in metrics:
                res['gaussian'].append(self._gaussian_fit(x=best, m=i, v=j))

        res['cluster'] = self._cluster_fit()

        return res

    def classify(self, classifier, train_atoms, test_atoms, x='max',
                 metrics=['cdf', 'optimistic']):
        """Classify ranked predictions based on acquisition function.

        Parameters
        ----------
        classifier : func
            User defined function to classify an atoms object.
        train_atoms : list
            List of atoms objects from training data upon which to base
            classification.
        test_atoms : list
            List of atoms objects from test data upon which to base
            classification.

        Returns
        -------
        res : dict
            A dictionary of lists containg the fitness of each test point for
            the different acquisition functions.
        """
        # Create dictionary with a list for each acquisition function.
        res = {}
        for key in metrics:
            res.update({key: []})
        best = defaultdict(list)

        # start by classifying the training data.
        for i, a in enumerate(train_atoms):
            c = classifier(a)
            best[c].append(self.targets[i])

        # Calcuate fitness based on acquisition functions.
        for i, a in enumerate(test_atoms):
            c = classifier(a)
            if c in best:
                # Select a fitness reference.
                if x == 'max':
                    b = max(self.targets)
                elif x == 'min':
                    b = min(self.targets)
                elif isinstance(x, float):
                    b = x
                p = self.predictions[i]
                u = self.uncertainty[i]
                res['cdf'].append(self._cdf_fit(x=b, m=p, v=u))
                res['optimistic'].append(self._optimistic_fit(x=b, m=p, v=u))
                if 'gaussian' in metrics:
                    res['gaussian'].append(self._gaussian_fit(x=b, m=p, v=u))
            else:
                res['cdf'].append(float('inf'))
                res['optimistic'].append(float('inf'))
                if 'gaussian' in metrics:
                    res['gaussian'].append(float('inf'))
        return res

    def _cdf_fit(self, x, m, v):
        """Calculate the cumulative distribution function.

        Parameters
        ----------
        x : float
            Known value.
        m : float
            Predicted mean.
        v : float
            Variance on prediction.
        """
        cdf = 0.5 * (1 + erf((m - x) / np.sqrt(2 * v ** 2)))

        return cdf

    def _optimistic_fit(self, x, m, v):
        """Find predictions that will optimistically lead to progress.

        Parameters
        ----------
        x : float
            Known value.
        m : float
            Predicted mean.
        v : float
            Variance on prediction.
        """
        a = m + v - x

        return a

    def _gaussian_fit(self, x, m, v):
        """Find predictions with highest probability at x.

        This acquisition function assumes a gaussian posterior.

        Parameters
        ----------
        x : float
            Known value.
        m : float
            Predicted mean.
        v : float
            Variance on prediction.
        """
        return np.exp(-np.abs(m - x)/(2.*v**2))

    def _cluster_fit(self):
        """Penalize test points that are too clustered."""
        fit = []

        cf = cluster_features(
            train_matrix=self.train_features, train_target=self.targets,
            k=self.k, test_matrix=self.test_features,
            test_target=self.predictions
            )

        train_count = Counter(cf['train_order'])

        for i, c in enumerate(cf['test_order']):
            fit.append(self.predictions[i] / train_count[c])

        return fit
