"""Functions to perform some natural selection."""
import numpy as np


def population_reduction(pop, fit, population_size):
    """Method to reduce population size to constant.

    Parameters
    ----------
    pop : list
        Extended population.
    fit : list
        Extended fitness assignment.
    population_size : int
        The population size.
    pareto : bool
        Flag to specify whether search is for Pareto optimal set.

    Returns
    -------
    population : list
        The population after natural selection.
    fitness : list
        The fitness for the current population.
    """
    # Combine parameters and sort.
    global_details = [[i, j] for i, j in zip(pop, fit)]
    global_details.sort(key=lambda x: float(x[1]), reverse=True)
    global_details = np.asarray(global_details)

    population = global_details[:population_size, 0].tolist()
    fitness = global_details[:population_size, 1].tolist()

    return population, fitness

    # Reinitialize everything as empty list.
    population, fitness = [], []

    # Fill the lists with current best candidates.
    index = 0
    while len(population) < population_size:
        population.append(global_details[index][0])
        fitness.append(global_details[index][1])
        index += 1

    return population, fitness


def remove_duplicates(population, fitness, accuracy):
    """Function to delete duplicate candidates based on fitness.

    Parameters
    ----------
    population : array
        The current population.
    fitness : array
        The fitness for the current population.
    accuracy : int
        Number of decimal places to include when finding unique.

    Returns
    -------
    population : list
        The population after duplicates deleted.
    fitness : list
        The fitness for the population after duplicates deleted.
    """
    fitness_round = np.round(fitness, accuracy)
    unique = np.unique(fitness_round)

    duplicate = []

    for index in range(len(fitness_round)):
        if fitness_round[index] in unique:
            unique = np.delete(
                unique, np.where(unique == fitness_round[index])[0][0])
        else:
            duplicate.append(index)

    population = np.delete(population, duplicate, axis=0)
    fitness = np.delete(fitness, duplicate, axis=0)

    return population, fitness
