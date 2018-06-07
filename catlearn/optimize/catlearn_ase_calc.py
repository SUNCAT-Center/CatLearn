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
                 settings=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.trained_process = trained_process
        self.ml_calc = ml_calc
        self.step = finite_step
        self.n_images = settings['n_images']
        self.max_step = settings['max_step']
        self.a_crit = settings['a_const']
        self.c_crit = settings['c_const']
        self.ind_constraints = settings['ind_constraints']
        self.all_pred_images = settings['all_pred_images']

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):

        # Atoms:
        self.atoms = atoms

        predictions = 0.0
        pred_value = 0.0
        forces = 0.0
        gradients = 0.0
        energy = 0.0

        # Previous paths of the atom i (for penalty function).

        all_prev_pos_label = []
        for i in self.all_pred_images:
            if i.info['label'] == atoms.info['label']:
            ######### Under test: ###################
            # if (i.info['label'] == atoms.info['label']) or (i.info['label']
            # == atoms.info['label']-1) or (i.info['label'] == atoms.info[
            # 'label']+1):
            ######### Under test: ###################
                if i.info['accepted_path'] is True:
                    tmp = i.get_positions().flatten()
                    all_prev_pos_label.append(tmp)
        all_prev_pos_label = apply_mask_ase_constraints(
        list_to_mask=all_prev_pos_label, mask_index=self.ind_constraints)[1]


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
                    d_atom_atom = distance.sqeuclidean(test[atom],
                    closest_train[atom])
                    if d_atom_atom >= max_step:
                        p_i = 0.0
                        a_const = self.a_crit
                        c_const = self.c_crit
                        d_const = 1.0
                        p_i = (a_const * ((d_atom_atom-max_step)**2)) / (c_const*(
                        d_atom_atom-max_step) + d_const)
                    if d_atom_atom < max_step:
                        p_i = 0.0
                    penalty_max += p_i

            pred_value = pred_value[0][0] + penalty_max

            return pred_value

        self.energy  = 0.0

        Calculator.calculate(self, atoms, properties, system_changes)

        pos_flatten = self.atoms.get_positions().flatten()

        test = apply_mask_ase_constraints(list_to_mask=[pos_flatten],
                                          mask_index=self.ind_constraints)[1]

        # Get energy:

        energy = pred_energy_test(test=test)

        # Get forces:

        gradients = np.zeros(len(pos_flatten))
        for i in range(len(self.ind_constraints)):
            index_force = self.ind_constraints[i]
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
