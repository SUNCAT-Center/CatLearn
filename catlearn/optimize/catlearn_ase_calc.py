import numpy as np
from catlearn.optimize.constraints import *
from catlearn.optimize.convert import *
from scipy.spatial import distance
from ase.calculators.calculator import Calculator, all_changes
from catlearn.optimize.penalty_atoms import *
from catlearn.utilities.penalty_functions import PenaltyFunctions
# from catlearn.regression.acquisition_functions import AcquisitionFunctions
from ase.io import read, write
from ase.data import covalent_radii

class CatLearn_ASE(Calculator):
    """CatLearn/ASE calculator.

    """
    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, trained_process=None, ml_calc=None, finite_step=1e-5,
                 n_images=None, max_step=None,
                 c_crit_penalty=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.trained_process = trained_process
        self.ml_calc = ml_calc
        self.step = finite_step
        self.n_images = n_images
        self.max_step = max_step
        self.c_crit = c_crit_penalty

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):

        # Atoms:
        self.atoms = atoms

        # Mask test (constraints):
        ind_constraints = create_mask_ase_constraints(ini=atoms,
            constraints=atoms._constraints)

        predictions = 0.0
        pred_value = 0.0
        forces = 0.0
        gradients = 0.0
        energy = 0.0

        # Previous paths of the atom i (for penalty function).

        prev_atoms = read('accepted_paths.traj',':')
        all_prev_pos_label = []
        for i in prev_atoms:
            if i.info['label'] == atoms.info['label']:
                tmp = i.get_positions().flatten()
                all_prev_pos_label.append(tmp)
        all_prev_pos_label = apply_mask_ase_constraints(
        list_to_mask=all_prev_pos_label, mask_index=ind_constraints)[1]


        def pred_energy_test(test, atoms=self.atoms, ml_calc=self.ml_calc,
                                  trained_process=self.trained_process,
                                  max_step=self.max_step,
                                  all_prev_pos_label=all_prev_pos_label):
            predictions = 0.0
            pred_value = 0.0
            penalty_max = 0.0
            predictions = ml_calc.get_predictions(trained_process,
                                                  test_data=test[0])
            pred_mean = predictions['pred_mean']
            pred_value = pred_mean

            # Penalize the predicted function:

            # Penalty too far atoms:
            if max_step is not None:
                d_test_list_train = distance.cdist([test[0]],
                                                   all_prev_pos_label,
                                                   'euclidean')
                closest_train = (all_prev_pos_label[np.argmin(d_test_list_train)])
                test = array_to_atoms(test[0])
                closest_train = array_to_atoms(closest_train)
                penalty_max = 0.0
                for atom in range(len(test)):
                    d_atom_atom = distance.euclidean(test[atom], closest_train[atom])
                    if d_atom_atom >= max_step:
                        p_i = 0.0
                        ##### Under test ##################
                        a_const = 100.0
                        c_const = 20.0
                        ##### Under test ##################
                        d_const = 10.0
                        p_i = (a_const * ((d_atom_atom-max_step)**2)) / (c_const*(
                        d_atom_atom-max_step) + d_const)
                    if d_atom_atom < max_step:
                        p_i = 0.0
                    penalty_max += p_i

            # Penalty too close atoms:
            ############# Under test ######################################
            # pos_atoms = atoms.get_positions()
            # cov_radii_atoms = covalent_radii[atoms.get_atomic_numbers()]/2
            #
            # penalty_min = 0.0
            # for i in range(0, len(pos_atoms)):
            #     for j in range(i+1, len(pos_atoms)-1):
            #         d_i_j = distance.euclidean(pos_atoms[i], pos_atoms[j])
            #         sum_of_radii = cov_radii_atoms[i] + cov_radii_atoms[j]
            #         if d_i_j <= sum_of_radii:
            #             p_min_i = 0.0
            #             a_c = 10.0
            #             c_c = 2.0
            #             d_c = 1.0
            #             p_min_i = (a_c * ((d_i_j-sum_of_radii)**2)) / (c_c*(
            #                        d_i_j-sum_of_radii) + d_c)
            #         if d_i_j > sum_of_radii:
            #             p_min_i = 0.0
            #         penalty_min += p_min_i
            # if penalty_min != 0.0:
            #     print('Stopped because atoms are too close', penalty_min)
            #     from ase.visualize import view
            #
            #     view(atoms)
            #     exit()
            # print(penalty_min)
            ############# Under test ######################################

            pred_value = pred_value[0][0] + penalty_max #+ penalty_min

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
            pos[0][i] = pos_flatten[index_force] + 2.0 * self.step
            f_pos2 = pred_energy_test(test=pos)
            pos = test.copy()
            pos[0][i] = pos_flatten[index_force] - 2.0 * self.step
            f_neg2 = pred_energy_test(test=pos)
            gradients[index_force] = (f_neg2 - 8.0 * f_neg + 8.0 * f_pos -
            f_pos2)/ (12.0 * self.step)
            # gradients[index_force] = (-f_neg + f_pos)/ (2 * self.step)

        forces = np.reshape(-gradients, (self.atoms.get_number_of_atoms(), 3))

        # Results:
        self.results['energy'] = energy
        self.results['forces'] = forces
