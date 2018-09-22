import numpy as np
from catlearn.optimize.constraints import apply_mask
from ase.calculators.calculator import Calculator, all_changes
from scipy.optimize import *
import copy


class CatLearnASE(Calculator):

    """Artificial CatLearn/ASE calculator.
    """

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, gp, index_constraints,
                 finite_step=1e-5, **kwargs):

        Calculator.__init__(self, **kwargs)

        self.gp = gp
        self.fs = finite_step
        self.ind_constraints = index_constraints

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):

        # Atoms object.
        self.atoms = atoms

        def pred_energy_test(test, gp=self.gp):

            # Get predictions.
            predictions = gp.predict(test_fp=test)
            return predictions['prediction'][0][0]

        Calculator.calculate(self, atoms, properties, system_changes)

        pos_flatten = self.atoms.get_positions().flatten()

        test_point = apply_mask(list_to_mask=[pos_flatten],
                                mask_index=self.ind_constraints)[1]

        # Get energy.
        energy = pred_energy_test(test=test_point)

        # Get forces:
        gradients = np.zeros(len(pos_flatten))
        for i in range(len(self.ind_constraints)):
            index_force = self.ind_constraints[i]
            pos = copy.deepcopy(test_point)
            pos[0][i] = pos_flatten[index_force] + self.fs
            f_pos = pred_energy_test(test=pos)
            pos = copy.deepcopy(test_point)
            pos[0][i] = pos_flatten[index_force] - self.fs
            f_neg = pred_energy_test(test=pos)
            gradients[index_force] = (-f_neg + f_pos) / (2.0 * self.fs)

        forces = np.reshape(-gradients, (self.atoms.get_number_of_atoms(), 3))

        # Results:
        self.results['energy'] = energy
        self.results['forces'] = forces


def predicted_energy_test(x0, gp):
    return gp.predict(test_fp=[x0])['prediction'][0][0]


def optimize_ml_using_scipy(x0, gp, ml_algo):

    args = (gp, )

    if ml_algo == 'Powell':
        result_min = fmin_powell(func=predicted_energy_test, x0=x0,
                                 args=args, maxiter=None, xtol=1e-12,
                                 full_output=False, disp=False)
        interesting_point = result_min

    if ml_algo == 'sBFGS':
        result_min = fmin_bfgs(f=predicted_energy_test, x0=x0,
                               args=args, disp=False, full_output=False,
                               gtol=1e-6)
        interesting_point = result_min

    if ml_algo == 'L-BFGS-B':
        result_min = fmin_l_bfgs_b(func=predicted_energy_test, x0=x0,
                                   approx_grad=True, args=args, disp=False,
                                   pgtol= 1e-8, epsilon=1e-6)
        interesting_point = result_min[0]

    if ml_algo == 'CG':
        result_min = fmin_cg(f=predicted_energy_test, x0=x0, args=args,
                             disp=False, full_output=True, retall=True,
                             gtol=1e-6)
        interesting_point = result_min[-1][-1]

    if ml_algo == 'Nelder-Mead':
        result_min = fmin(func=predicted_energy_test, x0=x0, args=args,
                          disp=False, full_output=True, retall=True,
                          xtol=1e-8, ftol=1e-8)
        interesting_point = result_min[-1][-1]

    return interesting_point
