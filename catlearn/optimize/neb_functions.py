from catlearn.optimize.constraints import *
from catlearn.optimize.catlearn_ase_calc import CatLearn_ASE
import copy
from ase.io import read

def train_ml_process(list_train, list_targets, list_gradients,
                     index_constraints, ml_calculator, scaling_targets):
    """Trains a machine learning process.

    Parameters (self):

    Parameters
    ----------
    list_train : list of positions (in Cartesian).
    list_targets : list of energies.
    list_gradients : list of gradients.
    index_constraints : list of constraints constraints generated
                          previously. In order to 'hide' fixed atoms to the ML
                          algorithm we create a constraints mask. This
                          allows to reduce the size of the training
                          features (avoids creating a large covariance matrix).
    Returns
    --------
    dictionary containing:
        scaling_targets : scaling for the energies of the training set.
        trained_process : trained process ready for testing.
        ml_calc : returns the ML calculator (if changes have been made,
              e.g. hyperparamter optimization).


    """

    if index_constraints is not None:
        list_train = apply_mask_ase_constraints(
                                   list_to_mask=list_train,
                                   mask_index=index_constraints)[1]
        list_gradients = \
                                   apply_mask_ase_constraints(
                                   list_to_mask=list_gradients,
                                   mask_index=index_constraints)[1]

    # Scale energies:

    list_targets = list_targets - scaling_targets

    trained_process = ml_calculator.train_process(
            train_data=list_train,
            target_data=list_targets,
            gradients_data=list_gradients)

    if ml_calculator.__dict__['opt_hyperparam'] is True:
        ml_calculator.opt_hyperparameters()

    return {'ml_calc':ml_calculator, 'trained_process': trained_process}

def create_ml_neb(is_endpoint, fs_endpoint, images_interpolation,
                  n_images, constraints, index_constraints, trained_process, \
                  ml_calculator, scaling_targets, iteration):


    # End-points of the NEB path:
    s_guess_ml = copy.deepcopy(is_endpoint)
    f_guess_ml = copy.deepcopy(fs_endpoint)

    # Create ML NEB path:
    imgs = [s_guess_ml]

    # Scale energies (initial):
    imgs[0].__dict__['_calc'].__dict__['results']['energy'] = \
    imgs[0].__dict__['_calc'].__dict__['results']['energy'] - scaling_targets

    # Append labels, uncertainty and iter to the first end-point:
    imgs[0].info['label'] = 0
    imgs[0].info['uncertainty'] = 0.0
    imgs[0].info['iteration'] = iteration

    for i in range(1, n_images-1):
        image = s_guess_ml.copy()
        image.info['label'] = i
        image.info['uncertainty'] = 0.0
        image.info['iteration'] = iteration
        image.set_calculator(CatLearn_ASE(trained_process=trained_process,
                                     ml_calc=ml_calculator,
                                     index_constraints = index_constraints
                                     ))
        if images_interpolation is not None:
            image.set_positions(images_interpolation[i].get_positions())
        image.set_constraint(constraints)
        imgs.append(image)

    # Scale energies (final):
    imgs.append(f_guess_ml)
    imgs[-1].__dict__['_calc'].__dict__['results']['energy'] = \
    imgs[-1].__dict__['_calc'].__dict__['results']['energy'] - scaling_targets

    # Append labels, uncertainty and iter to the last end-point:
    imgs[-1].info['label'] = n_images
    imgs[-1].info['uncertainty'] = 0.0
    imgs[-1].info['iteration'] = iteration

    return imgs


