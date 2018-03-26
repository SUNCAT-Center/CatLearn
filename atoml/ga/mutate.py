"""Define some mutation functions."""
import copy
import numpy as np


def random_permutation(parent_one):
    """Perform a random permutation on a parameter block.

    Parameters
    ----------
    parent_one : list
        List of params for first parent.
    mut_op : string
        String of operator for mutation.
    """
    p1 = copy.deepcopy(parent_one)
    index = np.random.randint(len(parent_one))
    if p1[index] == 1.:
        p1[index] = 0.
    else:
        p1[index] = 1.

    return p1
