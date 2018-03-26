"""Cut and splice mating function."""
import numpy as np


def cut_and_splice(parent_one, parent_two, size='random'):
    """Perform cut_and_splice between two parents.

    Parameters
    ----------
    parent_one : list
        List of params for first parent.
    parent_two : list
        List of params for second parent.
    size : str
        Define how to choose size of each cut.
    """
    if size is 'random':
        cut_point = np.random.randint(1, len(parent_one), 1)[0]
    offspring = np.concatenate((parent_one[:cut_point],
                                parent_two[cut_point:]))

    return offspring
