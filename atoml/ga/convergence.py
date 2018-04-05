"""Functions to check for convergence in the GA."""
import numpy as np


class Convergence(object):
    """Class to check convergence."""

    def __init__(self):
        """Initialize the class."""
        self.fitness = []
        self.previous = float('-inf')
        self.count = 0

    def no_progress(self, fitness, repeat):
        """Convergence based on a lack of any progress in the search.

        Parameters
        ----------
        fitness : array
            A List of fitnesses from the search.
        repeat : int
            Number of repeat generations with no progress.

        Returns
        -------
        converged : bool
            True if convergence has been reached, False otherwise.
        """
        # Work out whether the search has progressed.
        if np.max(fitness) > self.previous:
            self.previous = np.max(fitness)
            self.count = 0
        else:
            self.count += 1

        # Check if the convergence count has been reached.
        if self.count > repeat:
            return True

        return False

    def stagnation(self, fitness, repeat):
        """Convergence based on a stagnation of the population.

        Parameters
        ----------
        fitness : array
            A List of fitnesses from the search.
        repeat : int
            Number of repeat generations with no progress.

        Returns
        -------
        converged : bool
            True if convergence has been reached, False otherwise.
        """
        # Work out whether the search has progressed.
        if (np.min(fitness), np.max(fitness)) != self.previous:
            self.previous = (np.min(fitness), np.max(fitness))
            self.count = 0
        else:
            self.count += 1

        # Check if the convergence count has been reached.
        if self.count > repeat:
            return True

        return False
