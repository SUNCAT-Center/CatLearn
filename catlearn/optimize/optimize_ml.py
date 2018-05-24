import numpy as np
# from catlearn.regression.acquisition_functions import AcquisitionFunctions
from catlearn.utilities.penalty_functions import PenaltyFunctions
from scipy.optimize import *
from catlearn.optimize.io import *
from catlearn.optimize.penalty_atoms import *


def predicted_energy_test(test, ml_calc, trained_process, min_step, max_step,
    acq_fun, mode, list_train, list_targets, ase):

        """Function that returns the value of the predicted mean for a given
        test point. This function can be penalised w.r.t. to the distance of
        the test and the previously trained points (optional). If an
        acquisition function was set (optional), this function will
        return the score of the acquisition function.

        Parameters
        ----------
        test : array
            Test point. This point will be tested in the ML in order to get
            a predicted value.
        train : array
            List of trained points. Fingerprint.
        target : array
            List of target values.

        Returns
        -------
        pred_value : float
            The prediction (or score, when using acquisition functions) for
            the test data.

        """
        predictions = 0
        pred_value = 0
        uncertainty = 0

        # Get predicted mean and uncertainty:

        predictions = ml_calc.get_predictions(trained_process,
        test_data=test)
        pred_mean = predictions['pred_mean']

        if acq_fun is not None:
            print('Work in progress.'), exit()
            # uncertainty = predictions['uncertainty']

        pred_value = pred_mean

        # Penalize the predicted function (optional):

        if max_step or min_step is not None:
            penalty = PenaltyFunctions(train_features=list_train,
                                       test_features=[test])
        if max_step is not None:
            if ase is False:
                penalty_too_far = penalty.penalty_far(c_max_crit=1e2,
                d_max_crit=max_step)
            if ase is True:
                if (len(test) % 3) == 0:
                    penalty_too_far = penalty_too_far_atoms_v2(
                                                 list_train=list_train,
                                                 test=test, max_step=max_step)
                if (len(test) % 3) != 0:
                    penalty_too_far = penalty.penalty_far(c_max_crit=1e2,
                                                          d_max_crit=max_step)

            pred_value = pred_mean.copy() + penalty_too_far
        if min_step is not None:
            penalty_too_close = penalty.penalty_close(c_min_crit=1.0,
                                                      d_min_crit=min_step)
            pred_value = pred_mean.copy() + penalty_too_close

        # Pass data to the acquisition function (optional):

        if acq_fun is not None:
            print('Work in progress.'), exit()
            # acq = AcquisitionFunctions(objective=mode, kappa=2.5)
            # pred_value = -acq.rank(predictions=pred_value,
            #                        uncertainty=uncertainty,
            #                        targets=list_targets
            #                        )[acq_fun]

        if mode == 'min':
            return pred_value[0][0]  # For minimization problems.

        if mode == 'max':
            return -pred_value[0][0]  # For maximization problems.


def optimize_ml_using_scipy(self, x0):

    args = (self.ml_calc, self.trained_process, self.min_step, self.max_step,
            self.acq_fun, self.mode, self.list_train, self.list_targets,
            self.ase)

    if self.ml_algo == 'Powell':
        result_min = fmin_powell(func=predicted_energy_test, x0=x0,
                                 args=args, maxiter=None,
                                 full_output=True, disp=False)
        self.interesting_point = result_min[0]
        self.ml_f_min_pred_mean = result_min[1]
        self.ml_feval_pred_mean = result_min[4]
        if result_min[5] == 0:
            self.ml_convergence = 'True'
        if result_min[5] == 1:
            self.ml_convergence = 'False. Max. feval exceed.'
        if result_min[5] == 2:
            self.ml_convergence = 'False. Max. iter. exceed'

    if self.ml_algo == 'BFGS': # Round 10
        result_min = fmin_bfgs(f=predicted_energy_test, x0=x0,
                               args=args, disp=False, full_output=True,
                               gtol=1e-6)
        self.interesting_point = result_min[0]
        self.ml_f_min_pred_mean = result_min[1]
        self.ml_feval_pred_mean = result_min[4]
        if result_min[6] == 0:
            self.ml_convergence = 'True'
        if result_min[6] == 1:
            self.ml_convergence = 'False. Max iter. exceed.'
        if result_min[6] == 2:
            self.ml_convergence = 'False. Grad. not changing.'

    if self.ml_algo == 'L-BFGS-B': # Round 10
        result_min = fmin_l_bfgs_b(func=predicted_energy_test, x0=x0,
                                   approx_grad=True, args=args, disp=False,
                                   pgtol= 1e-6)

        self.interesting_point = result_min[0]
        self.ml_f_min_pred_mean = result_min[1]
        self.ml_feval_pred_mean = result_min[2]['funcalls']
        if result_min[2]['warnflag'] == 0:
            self.ml_convergence = 'True'
        if result_min[2]['warnflag'] == 1:
            self.ml_convergence = 'False. Max iter. exceed.'
        if result_min[2]['warnflag'] == 2:
            self.ml_convergence = 'False. Stopped.'

    if self.ml_algo == 'CG': # Round 10
        result_min = fmin_cg(f=predicted_energy_test, x0=x0, args=args,
                             disp=False, full_output=True, retall=True,
                             gtol=1e-6)
        self.interesting_point = result_min[-1][-1]
        self.ml_f_min_pred_mean = result_min[1]
        self.ml_feval_pred_mean = result_min[2]
        if result_min[4] == 0:
            self.ml_convergence = 'True'
        if result_min[4] == 1:
            self.ml_convergence = 'False. Max iter. exceed.'
        if result_min[4] == 2:
            self.ml_convergence = 'False. Grad. not changing.'

    if self.ml_algo == 'Nelder-Mead': # Round 10
        result_min = fmin(func=predicted_energy_test, x0=x0, args=args,
                          disp=False, full_output=True, retall=True,
                          xtol=1e-8, ftol=1e-8)
        self.interesting_point = result_min[-1][-1]
        self.ml_f_min_pred_mean = result_min[1]
        self.ml_feval_pred_mean = result_min[3]
        if result_min[4] == 0:
            self.ml_convergence = 'True'
        if result_min[4] == 1:
            self.ml_convergence = 'False. Max feval exceed.'
        if result_min[4] == 2:
            self.ml_convergence = 'False. Max iter exceed.'

    return self.interesting_point
