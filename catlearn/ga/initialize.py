"""Function to initialize a population."""
import numpy as np


def initialize_population(pop_size, dimension, dmax=None):
    """Generate a random starting population.

    Parameters
    ----------
    pop_size : int
        Population size.
    d_param : int
        Dimension of parameters in model.
    """
    pop = np.zeros((pop_size, dimension))

    # Number of active features.
    if dmax is None:
        n_active = np.arange(dimension)
        dmax = dimension
    else:
        n_active = np.arange(dmax)

    for ind in range(pop_size):
        new_param = np.random.choice(
                n_active, np.random.randint(dmax) + 1, replace=False)
        pop[ind][new_param] = 1

    return pop
