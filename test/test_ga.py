"""Test for the GA module."""
import unittest
import random
import numpy as np

from atoml.ga import GeneticAlgorithm
from atoml.regression import RidgeRegression

from common import get_data


class TestGeneticAlgorithm(unittest.TestCase):
    """Class to test feature selection with a GA."""

    def ff(self, train_features, train_targets, test_features, test_targets):
        """Ridge regression predictions."""
        # Test ridge regression predictions.
        rr = RidgeRegression(cv='loocv')
        reg = rr.find_optimal_regularization(X=train_features, Y=train_targets)
        coef = rr.RR(X=train_features, Y=train_targets, omega2=reg)[0]

        # Test the model.
        sumd = 0.
        for tf, tt in zip(test_features, test_targets):
            p = (np.dot(coef, tf))
            sumd += (p - tt) ** 2
        error = (sumd / len(test_features)) ** 0.5

        return error

    def test_feature_selection(self):
        """Simple test case to make sure it doesn't crash."""
        train_features, train_targets, _, _ = get_data()
        train_features = train_features[:, :20]

        ga = GeneticAlgorithm(population_size=10,
                              fit_func=self.ff,
                              features=train_features,
                              targets=train_targets,
                              population=None)
        self.assertEqual(np.shape(ga.population), (10, 20))

        ga.search(50)
        self.assertTrue(len(ga.population) == 10)
        self.assertTrue(len(ga.fitness) == 10)


if __name__ == '__main__':
    unittest.main()
