"""GP acquisition functions."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.special import erf
from collections import defaultdict, Counter
from scipy.stats import norm
from .clustering import cluster_features

class AcquisitionFunctions(object):
    """Base class for acquisition functions."""

    def __init__(self, x='max', k=3, kappa=1.5):
        """Initialization of class.

        x : string
            The user defines the objective of the acq. function to minimize
            ('min') or maximize ('max').
        k : int
            Number of cluster to generate with clustering.
        kappa: int
            Constant that controls the explotation/exploration ratio in UCB.
        noise: float
            Small number must be added in the denominator for stability.
        """
        self.x = x
        self.k = k
        self.noise = 1e-6
        self.kappa = kappa

    def rank(self,targets, predictions, uncertainty, train_features=None,
             test_features=None, metrics=['cdf', 'optimistic','UCB', 'EI',
             'PI']):
        """Rank predictions based on acquisition function.

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

        Returns
        -------
        res : dict
            A dictionary of lists containg the fitness of each test point for
            the different acquisition functions.
        metrics : list
            list of strings. 
            Accepted values are 'cdf','ucb, 'ei', 'pi', 'optimistic' and
            'gaussian'.
        """
        # Create dictionary with a list for each acquisition function.
        self.targets = targets
        self.predictions = predictions
        self.uncertainty = uncertainty
        self.train_features = train_features
        self.test_features = test_features
        res = {}
        for key in metrics:
            res.update({key: []})
        # Select a fitness reference.
        if self.x == 'max':
            best = max(self.targets)
        elif self.x == 'min':
            best = min(self.targets)
        elif isinstance(self.x, float):
            best = self.x

        # Calculate fitness based on acquisition functions.
        res['cdf'].append(self._cdf_fit(x=best))
        res['optimistic'].append(self._optimistic_fit(x=best))
        res['UCB'].append(self._UCB())
        res['EI'].append(self._EI(x=best))
        res['PI'].append(self._PI(x=best))
        if 'gaussian' in metrics:
            res['gaussian'].append(self._gaussian_fit(x=best))
        if 'cluster' in metrics:
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

        # Start by classifying the training data.
        for i, a in enumerate(train_atoms):
            c = classifier(a)
            best[c].append(self.targets[i])

        # Calculate fitness based on acquisition functions.
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

    def _cdf_fit(self, x):
        """Calculate the cumulative distribution function.

        Parameters
        ----------
        x : float
            Known value.

        """
        cdf = 0.5 * (1 + erf((self.predictions - x) / np.sqrt(2 * self.uncertainty **
        2)))

        return cdf

    def _optimistic_fit(self, x):
        """Find predictions that will optimistically lead to progress.

        Parameters
        ----------
        x : float
            Known value.

        """
        a = self.predictions + self.uncertainty - x

        return a

    def _gaussian_fit(self, x):
        """Find predictions that have the highest probability at x,
        assuming a gaussian posterior.

        Parameters
        ----------
        x : float
            Known value.

        """
        return np.exp(-np.abs(self.predictions - x)/(2.*self.uncertainty**2))

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

    def _UCB(self):
        """
        Upper-confidence bound acq. function.

        Parameters
        ----------
        kappa : float
            Parameter that controls exploitation/exploration.
        """
        if self.x == 'max':
            return -(self.predictions - self.kappa * self.uncertainty)

        if self.x == 'min':
            return -self.predictions + self.kappa * self.uncertainty

    def _EI(self, x):
        """
        Expected improvement acq. function.

        Parameters
        ----------
        x : float
            Best known value.

        """
        if self.x == 'max':
            z = (self.predictions - x) / (self.uncertainty + self.noise)
            return (self.predictions - x) * norm.cdf(z) + self.uncertainty * norm.pdf(
            z)

        if self.x == 'min':
            z = (-self.predictions + x) / (self.uncertainty + self.noise)
            return -((self.predictions - x) * norm.cdf(z) -
            self.uncertainty * norm.pdf(z))

    def _PI(self, x):
        """
        Probability of improvement acq. function.
        Parameters
        ----------
        x : float
            Best known value.
        """
        if self.x == 'max':
            z = (self.predictions - x) / (self.uncertainty + self.noise)
            return norm.cdf(z)

        if self.x == 'min':
            z = -((self.predictions - x) / (self.uncertainty + self.noise))
            return norm.cdf(z)