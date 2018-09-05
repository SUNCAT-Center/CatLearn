import numpy as np
from catlearn.optimize.constraints import apply_mask_ase_constraints
from ase.calculators.calculator import Calculator, all_changes
from scipy.optimize import *
import copy


class CatLearnASE(Calculator):

    """Artificial CatLearn/ASE calculator.
    """

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, trained_process, ml_calc, index_constraints,
                 calc_uncertainty=False, finite_step=5e-4, kappa=0.0,
                 **kwargs):

        Calculator.__init__(self, **kwargs)

        self.trained_process = trained_process
        self.ml_calc = ml_calc
        self.fs = finite_step
        self.ind_constraints = index_constraints
        self.kappa = kappa
        self.calc_uncertainty = calc_uncertainty

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
            acq_val = post_mean
            unc = 0.0
            if self.calc_uncertainty is True:
                unc = predictions['uncertainty'][0]
                acq_val = post_mean + kappa * unc

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


def predicted_energy_test(test, ml_calc, trained_process):

        """Function that returns the value of the predicted mean for a given
        test point. This function can be penalised w.r.t. to the distance of
        the test and the previously trained points (optional).

        Parameters
        ----------
        test : array
            Test point. This point will be tested in the ML in order to get
            a predicted value.
        ml_calc : object
            Machine learning calculator.
        trained_process : object
            Includes the trained process.

        Returns
        -------
        pred_value : float
            Surrogate model prediction.

        """
        pred_value = 0

        # Get predicted mean.

        pred_value = ml_calc.get_predictions(trained_process,
        test_data=test)['pred_mean']
        return pred_value[0][0]  # For minimization problems.


def optimize_ml_using_scipy(x0, ml_calc, trained_process, ml_algo):

    args = (ml_calc, trained_process)

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
