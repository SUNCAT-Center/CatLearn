"""Test for the GA module."""
import unittest
import numpy as np

from atoml.ga import GeneticAlgorithm
from atoml.ga.predictors import (minimize_error, minimize_error_descriptors,
                                 minimize_error_time)
from atoml.ga.convergence import Convergence
from atoml.ga.io import read_data

from common import get_data


class TestGeneticAlgorithm(unittest.TestCase):
    """Class to test feature selection with a GA."""

    def test_generic_predictor(self):
        """Simple test case to make sure it doesn't crash."""
        train_features, train_targets, _, _ = get_data()
        train_features = train_features[:, :20]

        ga = GeneticAlgorithm(population_size=10,
                              fit_func=minimize_error,
                              features=train_features,
                              targets=train_targets,
                              population=None)
        self.assertEqual(np.shape(ga.population), (10, 20))

        ga.search(50)
        self.assertTrue(len(ga.population) == 10)
        self.assertTrue(len(ga.fitness) == 10)

    def test_pareto(self):
        """Simple test case to make sure it doesn't crash."""
        train_features, train_targets, _, _ = get_data()
        train_features = train_features[:, :20]

        ga = GeneticAlgorithm(population_size=10,
                              fit_func=minimize_error_descriptors,
                              features=train_features,
                              targets=train_targets,
                              population=None,
                              fitness_parameters=2)
        self.assertEqual(np.shape(ga.population), (10, 20))

        ga.search(50)
        self.assertTrue(len(ga.population) == 10)
        self.assertTrue(len(ga.fitness) == 10)

        ga = GeneticAlgorithm(population_size=10,
                              fit_func=minimize_error_time,
                              features=train_features,
                              targets=train_targets,
                              population=None,
                              fitness_parameters=2)
        self.assertEqual(np.shape(ga.population), (10, 20))

        ga.search(50)
        self.assertTrue(len(ga.population) == 10)
        self.assertTrue(len(ga.fitness) == 10)

    def test_convergence(self):
        """Simple test case to make sure it doesn't crash."""
        train_features, train_targets, _, _ = get_data()
        train_features = train_features[:, :20]

        conv = Convergence()
        conv = conv.stagnation

        ga = GeneticAlgorithm(population_size=10,
                              fit_func=minimize_error,
                              features=train_features,
                              targets=train_targets,
                              population=None)
        self.assertEqual(np.shape(ga.population), (10, 20))

        ga.search(50, convergence_operator=conv)
        self.assertTrue(len(ga.population) == 10)
        self.assertTrue(len(ga.fitness) == 10)

    def test_read_write(self):
        """Function to test reading an writing GA files."""
        train_features, train_targets, _, _ = get_data()
        train_features = train_features[:, :20]

        ga1 = GeneticAlgorithm(population_size=10,
                               fit_func=minimize_error,
                               features=train_features,
                               targets=train_targets,
                               population=None)
        self.assertEqual(np.shape(ga1.population), (10, 20))

        ga1.search(2, writefile='gaWrite.json')
        self.assertTrue(len(ga1.population) == 10)
        self.assertTrue(len(ga1.fitness) == 10)

        old_pop, _ = read_data('gaWrite.json')

        ga2 = GeneticAlgorithm(population_size=10,
                               fit_func=minimize_error,
                               features=train_features,
                               targets=train_targets,
                               population=old_pop)
        self.assertTrue(np.allclose(ga2.population, ga1.population))

        ga2.search(50)
        self.assertTrue(len(ga2.population) == 10)
        self.assertTrue(len(ga2.fitness) == 10)


if __name__ == '__main__':
    unittest.main()
