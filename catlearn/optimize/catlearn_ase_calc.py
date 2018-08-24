# @Version u1.0.4

import numpy as np
from catlearn.optimize.constraints import apply_mask_ase_constraints
from ase.calculators.calculator import Calculator, all_changes
import copy


class CatLearnASE(Calculator):

    """Artificial CatLearn/ASE calculator.
    """

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, trained_process, ml_calc, index_constraints,
                 finite_step=5e-4, kappa=0.0, **kwargs):

        Calculator.__init__(self, **kwargs)

        self.trained_process = trained_process
        self.ml_calc = ml_calc
        self.fs = finite_step
        self.ind_constraints = index_constraints
        self.kappa = kappa

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):

        # Atoms object.
        self.atoms = atoms

        def pred_energy_test(test, ml_calc=self.ml_calc,
                             trained_process=self.trained_process,
                             kappa=self.kappa):

            # Get predictions.
            predictions = ml_calc.get_predictions(trained_process,
                                                  test_data=test[0])

            post_mean = predictions['pred_mean'][0][0]
            unc = predictions['uncertainty'][0]
            acq_val = post_mean + (kappa * unc)
            return [acq_val, unc]

        Calculator.calculate(self, atoms, properties, system_changes)

        pos_flatten = self.atoms.get_positions().flatten()

        test_point = apply_mask_ase_constraints(
                                            list_to_mask=[pos_flatten],
                                            mask_index=self.ind_constraints)[1]

        # Get energy and uncertainty.
        energy, uncertainty = pred_energy_test(test=test_point)

        # Attach uncertainty to Atoms object.
        atoms.info['uncertainty'] = uncertainty

        # Get forces:
        gradients = np.zeros(len(pos_flatten))
        for i in range(len(self.ind_constraints)):
            index_force = self.ind_constraints[i]
            pos = copy.deepcopy(test_point)
            pos[0][i] = pos_flatten[index_force] + self.fs
            f_pos = pred_energy_test(test=pos)[0]
            pos = copy.deepcopy(test_point)
            pos[0][i] = pos_flatten[index_force] - self.fs
            f_neg = pred_energy_test(test=pos)[0]
            gradients[index_force] = (-f_neg + f_pos) / (2.0 * self.fs)

        forces = np.reshape(-gradients, (self.atoms.get_number_of_atoms(), 3))

        # Results:
        self.results['energy'] = energy
        self.results['forces'] = forces
