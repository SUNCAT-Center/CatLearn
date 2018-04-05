"""Functions to read and write GA data."""
import numpy as np
import json


def _write_data(writefile, population, fitness):
    """Funtion to save population and fitness.

    Parameters
    ----------
    writefile : str
        Name of the JSON file to write.
    """
    data = {
        'population': np.asarray(population).tolist(),
        'fitness': np.asarray(fitness).tolist()
    }
    with open(writefile, 'w') as file:
        json.dump(data, file)


def read_data(writefile):
    """Funtion to read population and fitness.

    Parameters
    ----------
    writefile : str
        Name of the JSON file to read.

    Returns
    -------
    population : array
        The population saved from a previous search.
    fitness : array
        The fitness associated with the saved population.
    """
    with open(writefile, 'r') as file:
        data = json.load(file)

    population = np.array(data['population'])
    fitness = np.array(data['fitness'])

    return population, fitness
