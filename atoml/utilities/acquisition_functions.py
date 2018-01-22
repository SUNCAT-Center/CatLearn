"""GP acquisition functions."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.special import erf
from collections import defaultdict


class AcquisitionFunctions(object):
    """Base class for acquisition functions."""

    def __init__(self, targets, predictions, uncertainty):
        """Initialization of class.

        Parameters
        ----------
        targets : list
            List of knowns.
        predictions : list
            List of predictions from the GP.
        uncertainty : list
            List of variance on the GP predictions.
        """
        self.targets = targets
        self.predictions = predictions
        self.uncertainty = uncertainty

    def rank(self):
        """Rank predictions based on acquisition function.

        Returns
        -------
        res : dict
            A dictionary of lists containg the fitness of each test point for
            the different acquisition functions.
        """
        res = {'cdf': [], 'optimistic': []}
        best = max(self.targets)

        # Calcuate fitness based on acquisition functions.
        for i, j in zip(self.predictions, self.uncertainty):
            res['cdf'].append(self._cdf_fit(x=best, m=i, v=j))
            res['optimistic'].append(self._optimistic_fit(x=best, m=i, v=j))

        return res

    def classify(self, classifier, train_atoms, test_atoms):
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
        res = {'cdf': [], 'optimistic': []}
        best = defaultdict(list)

        # start by classifying the training data.
        for i, a in enumerate(train_atoms):
            c = classifier(a)
            best[c].append(self.targets[i])

        # Calcuate fitness based on acquisition functions.
        for i, a in enumerate(test_atoms):
            c = classifier(a)
            if c in best:
                b = max(best[c])
                p = self.predictions[i]
                u = self.uncertainty[i]
                res['cdf'].append(self._cdf_fit(x=b, m=p, v=u))
                res['optimistic'].append(self._optimistic_fit(x=b, m=p, v=u))
            else:
                res['cdf'].append(float('inf'))
                res['optimistic'].append(float('inf'))

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
