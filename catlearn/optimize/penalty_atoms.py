import numpy as np
from scipy.spatial import distance
from catlearn.optimize.convert import *


def penalty_too_far_atoms(list_train, test, max_step, c_max_crit=1e2):
    d_test_list_train = distance.cdist([test], list_train, 'euclidean')
    closest_train = (list_train[np.argmin(d_test_list_train)])
    test = array_to_atoms(test)
    closest_train = array_to_atoms(closest_train)
    penalty = 0
    for atom in range(len(test)):
        d_atom_atom = distance.euclidean(test[atom], closest_train[atom])
        if d_atom_atom >= max_step:
            p_i = c_max_crit * (d_atom_atom-max_step)**2
        else:
            p_i = 0
        penalty += p_i
    return penalty


def penalty_too_far_atoms_v2(list_train, test, max_step, penalty_constant):
    d_test_list_train = distance.cdist([test], list_train, 'euclidean')
    closest_train = (list_train[np.argmin(d_test_list_train)])
    test = array_to_atoms(test)
    closest_train = array_to_atoms(closest_train)
    penalty = 0
    for atom in range(len(test)):
        d_atom_atom = distance.euclidean(test[atom], closest_train[atom])
        if d_atom_atom >= max_step:
            a_const = penalty_constant
            c_const = 2.0
            d_const = 1.0
            p_i = (a_const * ((d_atom_atom-max_step)**2)) / (c_const*np.abs(
            d_atom_atom-max_step) + d_const)
        else:
            p_i = 0
        penalty += p_i
    return penalty


def penalty_too_far(list_train, test, max_step=None, c_max_crit=1e2):
    """ Pass an array of test features and train features and
    returns an array of penalties due to 'too far distance'.
    This prevents to explore configurations that are unrealistic.

    Parameters
    ----------
    d_max_crit : float
        Critical distance.
    c_max_crit : float
        Constant for penalty minimum distance.
    penalty_max: array
        Array containing the penalty to add.
    """
    penalty_max = []
    for i in test:
        d_max = np.min(distance.cdist([i], list_train,'euclidean'))
        if d_max >= max_step:
            p = c_max_crit * (d_max-max_step)**2
        else:
            p = 0.0
        penalty_max.append(p)
    return penalty_max

# def penalty_atoms_too_close(test, min_dist, a_c=10.0):
#     test = array_to_atoms(test)
#     penalty = 0
#     for atom in range(len(test)):
#         d_atom_atom = distance.euclidean(atom, atom)
#         if d_atom_atom <= min_dist:
#             a_const = a_c
#             c_const = 2.0
#             d_const = 1.0
#             p_i = (a_const * ((d_atom_atom-min_dist)**2)) / (c_const*np.abs(
#             d_atom_atom-min_dist) + d_const)
#         else:
#             p_i = 0
#         penalty += p_i
#     return penalty

