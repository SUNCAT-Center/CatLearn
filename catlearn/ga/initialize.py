"""Function to initialize a population."""
import numpy as np


def initialize_population(pop_size, dimension):
    """Generate a random starting population.

    Parameters
    ----------
    pop_size : int
        Population size.
    d_param : int
        Dimension of parameters in model.
    """
    pop = np.ones((pop_size, dimension))
    index = np.arange(dimension)
    for ind in range(pop_size):
        new_param = np.random.choice(
            index, np.random.randint(dimension) + 1, replace=False)
        pop[ind][new_param] = 0.

    return pop
