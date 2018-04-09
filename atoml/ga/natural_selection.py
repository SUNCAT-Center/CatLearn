"""Functions to perform some natural selection."""


def population_reduction(pop, fit, population_size, pareto):
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

    Attributes
    ----------
    population : list
        The population after natural selection.
    fitness : list
        The fitness for the current population.
    """
    # Combine parameters and sort.
    global_details = [[i, j] for i, j in zip(pop, fit)]
    global_details.sort(key=lambda x: float(x[1]), reverse=True)

    # Reinitialize everything as empty list.
    population, fitness, unique_list = [], [], []

    # Fill the lists with current best candidates.
    for i in global_details:
        if len(population) < population_size:
            # Round to some tolerance to make sure unique fitness.
            if round(i[1], 5) not in unique_list and not pareto:
                population.append(i[0])
                fitness.append(i[1])
                unique_list.append(round(i[1], 5))
            else:
                population.append(i[0])
                fitness.append(i[1])
        else:
            break

    return population, fitness
