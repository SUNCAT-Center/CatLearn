from catlearn.optimize.constraints import *
from catlearn.optimize.catlearn_ase_calc import CatLearn_ASE
import copy
from ase.io import read

def train_ml_process(list_train, list_targets, list_gradients,
                     index_constraints, ml_calculator, scale_targets):
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
        scale : scaling for the energies of the training set.
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

    list_targets = list_targets - scale_targets

    trained_process = ml_calculator.train_process(
            train_data=list_train,
            target_data=list_targets,
            gradients_data=list_gradients)

    if ml_calculator.__dict__['opt_hyperparam']:
        ml_calculator.opt_hyperparameters()

    return {'ml_calc':ml_calculator, 'trained_process': trained_process}

def create_ml_neb(images_interpolation, trained_process, ml_calculator,
                  settings_neb_dict):



    # End-points of the NEB path:
    start_guess_ml = copy.deepcopy(settings_neb_dict['initial_endpoint'])
    final_guess_ml = copy.deepcopy(settings_neb_dict['final_endpoint'])

    # Create ML NEB path:
    images = [start_guess_ml]

    # Scale energies (initial):
    images[0].__dict__['_calc'].__dict__['results']['energy'] = \
    images[0].__dict__['_calc'].__dict__['results']['energy'] - \
    settings_neb_dict['scale_targets']

    # Append labels, uncertainty and iter to the first end-point:
    images[0].info['label'] = 0
    images[0].info['uncertainty'] = 0.0
    images[0].info['iteration'] = settings_neb_dict['iteration']
    images[0].info['scaling'] = settings_neb_dict['scale_targets']
    images[0].info['accepted_path'] = True

    for i in range(0, settings_neb_dict['n_images']-2):
        image = start_guess_ml.copy()
        image.info['label'] = i+1
        image.info['uncertainty'] = 0.0
        image.info['iteration'] = settings_neb_dict['iteration']
        image.info['scaling'] = settings_neb_dict['scale_targets']
        image.info['accepted_path'] = True
        image.set_calculator(CatLearn_ASE(trained_process=trained_process,
                                     ml_calc=ml_calculator,
                                     settings=settings_neb_dict))
        if images_interpolation is not None:
            image.set_positions(images_interpolation[i].get_positions())
        image.set_constraint(settings_neb_dict['constraints'])
        images.append(image)

    # Scale energies (final):
    images.append(final_guess_ml)
    images[-1].__dict__['_calc'].__dict__['results']['energy'] = \
    images[-1].__dict__['_calc'].__dict__['results']['energy'] - \
    settings_neb_dict['scale_targets']

    # Append labels, uncertainty and iter to the last end-point:
    images[-1].info['label'] = settings_neb_dict['n_images']
    images[-1].info['uncertainty'] = 0.0
    images[-1].info['iteration'] = settings_neb_dict['iteration']
    images[-1].info['scaling'] = settings_neb_dict['scale_targets']
    images[-1].info['accepted_path'] = True

    return images


