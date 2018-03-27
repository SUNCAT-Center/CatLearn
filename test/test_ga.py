"""Test for the GA module."""
import unittest
import random
import numpy as np

from atoml.ga import GeneticAlgorithm


class TestGeneticAlgorithm(unittest.TestCase):
    """Class to test feature selection with a GA."""

    def ff(self, x):
        """Some random fitness is returned."""
        return random.random()

    def test_feature_selection(self):
        """Simple test case to make sure it doesn't crash."""
        ga = GeneticAlgorithm(pop_size=10,
                              fit_func=self.ff,
                              dimension=20,
                              pop=None)
        self.assertEqual(np.shape(ga.pop), (10, 20))

        ga.search(50)
        self.assertTrue(len(ga.pop) == 10)
        self.assertTrue(len(ga.fitness) == 10)


if __name__ == '__main__':
    unittest.main()
