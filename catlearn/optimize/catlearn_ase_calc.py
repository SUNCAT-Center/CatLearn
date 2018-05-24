import numpy as np
from catlearn.optimize.constraints import *
from catlearn.optimize.convert import *
from scipy.spatial import distance
from ase.calculators.calculator import Calculator, all_changes
from catlearn.optimize.penalty_atoms import *
from catlearn.utilities.penalty_functions import PenaltyFunctions
# from catlearn.regression.acquisition_functions import AcquisitionFunctions
from ase.io import read, write
class CatLearn_ASE(Calculator):
    """CatLearn/ASE calculator.

    """
    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, trained_process=None, ml_calc=None, finite_step=5e-4,
                 n_images=None, max_step=None, penalty_mode=None,
                 c_crit_penalty=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.trained_process = trained_process
        self.ml_calc = ml_calc
        self.step = finite_step
        self.n_images = n_images
        self.max_step = max_step
        self.penalty_mode = penalty_mode
        self.c_crit = c_crit_penalty

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):

        # Mask test (constraints):
        ind_constraints = create_mask_ase_constraints(ini=atoms,
            constraints=atoms._constraints)

        predictions = 0
        pred_value = 0
        forces = 0
        gradients = 0
        energy = 0

        prev_atoms = read('all_pred_paths.traj',':')

        all_prev_positions_label_i = []
        for i in prev_atoms:
            if i.info['label'] == atoms.info['label']:
                tmp = i.get_positions().flatten()
                all_prev_positions_label_i.append(tmp)
        all_prev_positions_label_i = apply_mask_ase_constraints(
        list_to_mask=all_prev_positions_label_i, mask_index=ind_constraints)[1]

        prev_positions_label_i = []
        for i in prev_atoms[-self.n_images:]:
            if i.info['label'] == atoms.info['label']:
                tmp = i.get_positions().flatten()
                prev_positions_label_i.append(tmp)
        prev_positions_label_i = apply_mask_ase_constraints(
        list_to_mask=prev_positions_label_i, mask_index=ind_constraints)[1]

        all_prev_positions=[]
        for i in prev_atoms:
            tmp = i.get_positions().flatten()
            all_prev_positions.append(tmp)
        all_prev_positions = apply_mask_ase_constraints(
        list_to_mask=all_prev_positions, mask_index=ind_constraints)[1]

        last_prev_positions=[]
        for i in prev_atoms[-self.n_images:]:
            tmp = i.get_positions().flatten()
            last_prev_positions.append(tmp)
        last_prev_positions = apply_mask_ase_constraints(
        list_to_mask=last_prev_positions, mask_index=ind_constraints)[1]

        def pred_energy_test(test, ml_calc=self.ml_calc,
                                  trained_process=self.trained_process,
                                  max_step=self.max_step,
                                  c_crit = self.c_crit,
                                  all_prev_positions=all_prev_positions,
                                  prev_positions=last_prev_positions,
                                  prev_positions_label_i=prev_positions_label_i,
                                  all_prev_positions_label_i=all_prev_positions_label_i):
            predictions = 0
            pred_value = 0
            penalty_max = 0

            predictions = ml_calc.get_predictions(trained_process,
                                                  test_data=test[0])
            pred_mean = predictions['pred_mean']
            pred_value = pred_mean

            # Penalize the predicted function (optional):
            if max_step is not None:

                selec_mode = ['all_neb_images', 'last_neb_images',
                              'all_neb_atoms', 'all_neb_atoms_label',
                              'last_neb_atoms',
                              'all_neb_images_label', 'all_prev_train', 'all_neb_atoms_label_v2']
                if self.penalty_mode == 'all_neb_images':

                    penalty_max = penalty_too_far(list_train=all_prev_positions,
                                                  test=test,
                                                  max_step=self.max_step,
                                                  c_max_crit=c_crit)
                if self.penalty_mode == 'last_neb_images':
                    penalty_max = penalty_too_far(list_train=prev_positions,
                                                  test=test,
                                                  max_step=self.max_step,
                                                  c_max_crit=c_crit)
                if self.penalty_mode == 'all_neb_atoms':
                    penalty_max = penalty_too_far_atoms(
                                                list_train=all_prev_positions,
                                                  test=test[0],
                                                  max_step=self.max_step,
                                                  c_max_crit=c_crit)
                if self.penalty_mode == 'all_neb_atoms_label':
                    penalty_max = penalty_too_far_atoms(
                                                list_train=all_prev_positions_label_i,
                                                  test=test[0],
                                                  max_step=self.max_step,
                                                  c_max_crit=c_crit)
                if self.penalty_mode == 'all_neb_atoms_label_v2':
                    penalty_max = penalty_too_far_atoms_v2(
                                                list_train=all_prev_positions_label_i,
                                                  test=test[0],
                                                  max_step=self.max_step)
                if self.penalty_mode == 'last_neb_atoms':
                    penalty_max = penalty_too_far_atoms(
                                    list_train=prev_positions,
                                    test=test[0],
                                    max_step=self.max_step,
                                    c_max_crit=c_crit)
                if self.penalty_mode == 'last_neb_images_label':
                    penalty_max = penalty_too_far(
                                                list_train=prev_positions_label_i,
                                                  test=test,
                                                  max_step=self.max_step,
                                                  c_max_crit=c_crit)
                if self.penalty_mode == 'all_neb_images_label':
                    penalty_max = penalty_too_far(
                                                list_train=all_prev_positions_label_i,
                                                  test=test,
                                                  max_step=self.max_step,
                                                  c_max_crit=c_crit)


                if self.penalty_mode not in selec_mode:
                    msg = 'Error. Not a penalty mode.'
                    print(msg),exit()

                pred_value = pred_value[0] + penalty_max
                atoms.info['penalty_max'] = penalty_max
            return pred_value

        self.energy  = 0.0

        Calculator.calculate(self, atoms, properties, system_changes)

        pos_flatten = self.atoms.get_positions().flatten()

        test = apply_mask_ase_constraints(list_to_mask=[pos_flatten],
        mask_index=ind_constraints)[1]

        # Get energy:
        energy = pred_energy_test(test=test)

        # Get forces:

        gradients = np.zeros(len(pos_flatten))

        for i in range(len(ind_constraints)):
            index_force = ind_constraints[i]
            pos = test.copy()
            pos[0][i] = pos_flatten[index_force] + self.step
            f_pos = pred_energy_test(test=pos)
            pos = test.copy()
            pos[0][i] = pos_flatten[index_force] - self.step
            f_neg = pred_energy_test(test=pos)
            pos = test.copy()
            pos[0][i] = pos_flatten[index_force] + 2 * self.step
            f_pos2 = pred_energy_test(test=pos)
            pos = test.copy()
            pos[0][i] = pos_flatten[index_force] - 2 * self.step
            f_neg2 = pred_energy_test(test=pos)
            gradients[index_force] = (f_neg2 - 8 * f_neg + 8 * f_pos -
            f_pos2)/ (12 * self.step)
            # gradients[index_force] = (-f_neg + f_pos)/ (2 * self.step)

        forces = np.reshape(-gradients, (self.atoms.get_number_of_atoms(), 3))

        # Get penalty max:
        if atoms.info['penalty_max'] != 0:
            atoms.info['penalized'] = True

        # Results:
        self.results['energy'] = energy
        self.results['forces'] = forces
