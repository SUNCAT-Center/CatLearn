"""GP acquisition functions."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.special import erf


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
        """Rank predictions based on acquisition function."""
        res = {'cdf': [], 'optimistic': []}
        best = max(self.targets)

        for i, j in zip(self.predictions, self.uncertainty):
            res['cdf'].append(self._cdf_fit(x=best, m=i, v=j))
            res['optimistic'].append(self._optimistic_fit(x=best, m=i, v=j))

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
