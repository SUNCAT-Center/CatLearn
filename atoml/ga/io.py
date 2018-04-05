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
        'population': population.tolist(),
        'fitness': fitness.tolist()
    }
    with open(writefile, 'w') as file:
        json.dump(data, file)


def read_data(self, writefile):
    """Funtion to read population and fitness.

    Parameters
    ----------
    writefile : str
        Name of the JSON file to read.
    """
    with open(writefile, 'r') as file:
        data = json.load(file)

    population = np.array(data['population'])
    fitness = np.array(data['fitness'])

    return population, fitness
