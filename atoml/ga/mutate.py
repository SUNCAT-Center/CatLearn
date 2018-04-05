"""Define some mutation functions."""
import copy
import numpy as np


def random_permutation(parent_one):
    """Perform a random permutation on a parameter index.

    Parameters
    ----------
    parent_one : list
        List of params for first parent.

    Returns
    -------
    p1 : list
        Mutated parameter list based on the parent parameters provided.
    """
    p1 = copy.deepcopy(parent_one)
    index = np.random.randint(len(parent_one))
    if p1[index] == 1.:
        p1[index] = 0.
    else:
        p1[index] = 1.

    return p1


def probability_remove(parent_one):
    """A mutation that will remove features with a certain probability.

    Parameters
    ----------
    parent_one : list
        List of params for first parent.

    Returns
    -------
    p1 : list
        Mutated parameter list based on the parent parameters provided.
    """
    p1 = copy.deepcopy(parent_one)
    probability = np.random.random()
    for index in range(parent_one.shape[0]):
        if p1[index] == 1. and np.random.random() >= probability:
            p1[index] = 0.

    # Make sure not all features are removed.
    if len(p1[p1 == 0.]) == len(p1):
        index = np.random.randint(len(parent_one))
        p1[index] = 1.

    return p1


def probability_include(parent_one):
    """A mutation that will include features with a certain probability.

    Parameters
    ----------
    parent_one : list
        List of params for first parent.

    Returns
    -------
    p1 : list
        Mutated parameter list based on the parent parameters provided.
    """
    p1 = copy.deepcopy(parent_one)
    probability = np.random.random()
    for index in range(parent_one.shape[0]):
        if p1[index] == 0. and np.random.random() >= probability:
            p1[index] = 1.

    return p1
