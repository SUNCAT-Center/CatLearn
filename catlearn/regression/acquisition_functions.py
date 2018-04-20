"""GP acquisition functions."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.special import erf
from scipy.stats import norm
from collections import defaultdict, Counter

from catlearn.utilities.clustering import cluster_features


class AcquisitionFunctions(object):
    """Base class for acquisition functions."""

    def __init__(self, objective='max', k_means=3, kappa=1.5):
        """Initialization of class.

        objective : string
            The user defines the objective of the acq. function to minimize
            ('min') or maximize ('max').
        k_means : int
            Number of cluster to generate with clustering.
        kappa: int
            Constant that controls the explotation/exploration ratio in UCB.
        noise: float
            Small number must be added in the denominator for stability.
        """
        self.objective = objective
        self.k_means = k_means
        self.noise = 1e-6
        self.kappa = kappa

    def rank(self, targets, predictions, uncertainty, train_features=None,
             test_features=None, metrics=['cdf', 'optimistic', 'UCB', 'EI',
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
        metrics : list
            list of strings.
            Accepted values are 'cdf', 'UCB', 'EI', 'PI', 'optimistic' and
            'gaussian'.

        Returns
        -------
        res : dict
            A dictionary of lists containg the fitness of each test point for
            the different acquisition functions.

        """
        self.targets = targets
        self.predictions = predictions
        self.uncertainty = uncertainty
        self.train_features = train_features
        self.test_features = test_features

        # Create dictionary for each acquisition function.
        res = {}

        # Select a fitness reference.
        if self.objective == 'max':
            self.y_best = max(self.targets)
        elif self.objective == 'min':
            self.y_best = min(self.targets)
        elif isinstance(self.objective, float):
            self.y_best = self.objective

        # Calculate fitness based on acquisition functions.
        res['cdf'] = self._cdf_fit()
        res['optimistic'] = self._optimistic_fit()
        res['UCB'] = self._UCB()
        res['EI'] = self._EI()
        res['PI'] = self._PI()
        if 'gaussian' in metrics:
            res['gaussian'] = self._gaussian_fit()
        if 'cluster' in metrics:
            res['cluster'] = self._cluster_fit()

        return res

    def classify(self, classifier, train_atoms, test_atoms, targets,
                 predictions, uncertainty, train_features=None,
                 test_features=None, metrics=['cdf', 'optimistic', 'UCB', 'EI',
                                              'PI']):
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
        metrics : list
            list of strings.
            Accepted values are 'cdf', 'UCB', 'EI', 'PI', 'optimistic' and
            'gaussian'.

        Returns
        -------
        res : dict
            A dictionary of lists containg the fitness of each test point for
            the different acquisition functions.
        """
        # Start by classifying the training data.
        best = defaultdict(list)
        for i, a in enumerate(train_atoms):
            c = classifier(a)
            best[c].append(self.targets[i])

        for i in best:
            # Select a fitness reference.
            if self.objective == 'max':
                best[i] = max(best[i])
            elif self.objective == 'min':
                best[i] = min(best[i])
            elif isinstance(self.objective, float):
                best[i] = self.objective

        test = defaultdict(dict)
        params = ['index', 'atoms', 'predictions', 'uncertainty',
                  'train_features', 'test_features']
        # Calculate fitness based on acquisition functions.
        for i, a in enumerate(test_atoms):
            c = classifier(a)
            if c not in test:
                for key in params:
                    test[c].update({key: []})
            test[c]['index'].append(i)
            test[c]['atoms'].append(a)
            # test[c]['targets'].append(targets[i])
            test[c]['predictions'].append(predictions[i])
            test[c]['uncertainty'].append(uncertainty[i])
            test[c]['train_features'].append(train_features[i])
            test[c]['test_features'].append(test_features[i])

        tmp_res = defaultdict(dict)
        for i in test:
            tmp_res[i]['index'] = test[i]['index']
            self.predictions = np.asarray(test[i]['predictions'])
            self.uncertainty = np.asarray(test[i]['uncertainty'])
            self.train_features = np.asarray(test[i]['train_features'])
            self.test_features = np.asarray(test[i]['test_features'])
            tmp_res[i]['cdf'] = self._cdf_fit()
            tmp_res[i]['optimistic'] = self._optimistic_fit()
            tmp_res[i]['UCB'] = self._UCB()
            tmp_res[i]['EI'] = self._EI()
            tmp_res[i]['PI'] = self._PI()
            if 'gaussian' in metrics:
                tmp_res[i]['gaussian'] = self._gaussian_fit()
            if 'cluster' in metrics:
                raise NotImplemented

        res = {}
        size = len(test_atoms)
        res['cdf'] = np.zeros(size)
        res['optimistic'] = np.zeros(size)
        res['UCB'] = np.zeros(size)
        res['EI'] = np.zeros(size)
        res['PI'] = np.zeros(size)
        if 'gaussian' in metrics:
            res['gaussian'] = np.zeros(size)
        for i in tmp_res:
            for j, k in enumerate(tmp_res[i]['index']):
                res['cdf'][k] = tmp_res[i]['cdf'][j]
                res['optimistic'][k] = tmp_res[i]['optimistic'][j]
                res['UCB'][k] = tmp_res[i]['UCB'][j]
                res['EI'][k] = tmp_res[i]['EI'][j]
                res['PI'][k] = tmp_res[i]['PI'][j]
                if 'gaussian' in metrics:
                    res['gaussian'][k] = tmp_res[i]['gaussian'][j]

        return res

    def _cdf_fit(self):
        """Calculate the cumulative distribution function."""
        cdf = 0.5 * (
            1 + erf((self.predictions - self.y_best) / np.sqrt(
                2 * self.uncertainty ** 2)))

        return cdf

    def _optimistic_fit(self):
        """Find predictions that will optimistically lead to progress."""
        a = self.predictions + self.uncertainty - self.y_best
        return a

    def _gaussian_fit(self):
        """Find predictions that have the highest probability at x.

        This function assumes a gaussian posterior.
        """
        return np.exp(-np.abs(self.predictions - self.y_best) / (
            2. * self.uncertainty**2))

    def _cluster_fit(self):
        """Penalize test points that are too clustered."""
        fit = []

        cf = cluster_features(
            train_matrix=self.train_features, train_target=self.targets,
            k_means=self.k_means, test_matrix=self.test_features,
            test_target=self.predictions
        )

        train_count = Counter(cf['train_order'])

        for i, c in enumerate(cf['test_order']):
            fit.append(self.predictions[i] / train_count[c])

        return fit

    def _UCB(self):
        """Upper-confidence bound acq. function."""
        if self.objective == 'max':
            return -(self.predictions - self.kappa * self.uncertainty)

        if self.objective == 'min':
            return -self.predictions + self.kappa * self.uncertainty

    def _EI(self):
        """Expected improvement acq. function."""
        if self.objective == 'max':
            z = (self.predictions - self.y_best) / \
                (self.uncertainty + self.noise)
            return (self.predictions - self.y_best) * norm.cdf(z) + \
                self.uncertainty * norm.pdf(
                z)

        if self.objective == 'min':
            z = (-self.predictions + self.y_best) / \
                (self.uncertainty + self.noise)
            return -((self.predictions - self.y_best) * norm.cdf(z) -
                     self.uncertainty * norm.pdf(z))

    def _PI(self):
        """Probability of improvement acq. function."""
        if self.objective == 'max':
            z = (self.predictions - self.y_best) / \
                (self.uncertainty + self.noise)
            return norm.cdf(z)

        if self.objective == 'min':
            z = -((self.predictions - self.y_best) /
                  (self.uncertainty + self.noise))
            return norm.cdf(z)
