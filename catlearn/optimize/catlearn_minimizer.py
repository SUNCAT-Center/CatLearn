# @Version u2.5.0

from ase import Atoms
from ase.io.trajectory import TrajectoryWriter
from ase.optimize import *
from catlearn.optimize.warnings import *
from catlearn.optimize.ml_calculator import GPCalculator, train_ml_process
from catlearn.optimize.io import ase_traj_to_catlearn, print_info, array_to_ase
from catlearn.optimize.constraints import create_mask_ase_constraints, \
                                    unmask_geometry, apply_mask_ase_constraints

from ase.optimize.sciopt import *
from catlearn.optimize.get_real_values import eval_and_append, \
                                              get_energy_catlearn, \
                                              get_forces_catlearn
from catlearn.optimize.convergence import converged, get_fmax
from catlearn.optimize.catlearn_ase_calc import CatLearnASE
import os
import numpy as np
from catlearn.optimize.ml_optimize import optimize_ml_using_scipy

class CatLearnMin(object):

    def __init__(self, x0, ase_calc=None, ml_calc='SQE_sequential',
                 trajectory='catlearn_opt.traj'):

        """Optimization setup.

        Parameters
        ----------
        x0 : Atoms object or trajectory file in ASE format.
            Initial guess.
        ase_calc: ASE calculator object.
            When using ASE the user must pass an ASE calculator.
        ml_calc : Machine Learning calculator object.
            Machine Learning calculator (e.g. Gaussian Processes).
        trajectory: string
            Filename to store the output.
        """

        # General variables.
        self.filename = trajectory  # Remove extension if added.
        self.ml_calc = ml_calc
        self.iter = 0
        self.feval = 0
        self.fmax = 0.0
        self.min_iter = 0
        self.jac = True

        self.ase_calc = ase_calc

        assert x0 is not None, err_not_x0()

        if isinstance(x0, Atoms):
            self.start_mode = 'atoms'
            self.ase = True
            warning_using_ase()
            self.initial_structure = x0
            self.ase_ini = x0
            ase_calc_set = self.ase_calc
            self.ase_calc = self.ase_ini.get_calculator()
            if ase_calc_set is not None:
                self.ase_calc = ase_calc_set
            assert self.ase_calc, err_not_ase_calc_atoms()
            self.constraints = self.ase_ini._get_constraints()
            self.x0 = self.ase_ini.get_positions().flatten()
            self.num_atoms = self.ase_ini.get_number_of_atoms()
            self.formula = self.ase_ini.get_chemical_formula()
            self.list_train = [self.x0]
            self.list_targets = []
            self.list_gradients = []
            if len(self.constraints) < 0:
                self.constraints = None
            if self.constraints is not None:
                self.ind_mask_constr = create_mask_ase_constraints(
                                                self.ase_ini, self.constraints)

        if isinstance(x0, str):
            self.start_mode = 'trajectory'
            self.ase = True
            self.ase_calc = ase_calc
            self.i_ase_step = 'Previous calc.'
            warning_traj_detected()
            assert self.ase_calc, err_not_ase_calc_traj()
            trj = ase_traj_to_catlearn(traj_file=x0)
            self.list_train, self.list_targets, self.list_gradients, \
                trj_images, self.constraints, self.num_atoms = [
                    trj['list_train'], trj['list_targets'],
                    trj['list_gradients'], trj['images'],
                    trj['constraints'], trj['num_atoms']]
            for i in range(1, len(trj_images)):
                self.ase_ini = trj_images[i]
                molec_writer = TrajectoryWriter('./' + str(self.filename),
                                                mode='a')
                molec_writer.write(self.ase_ini)
            if len(self.constraints) < 0:
                self.constraints = None
            if self.constraints is not None:
                self.ind_mask_constr = create_mask_ase_constraints(
                    self.ase_ini, self.constraints)

        # Default kernel:

        if isinstance(self.ml_calc, str):
            implemented_kernels = ['SQE_static', 'SQE_isotropic', 'ARD_SQE',
                                   'SQE_anisotropic', 'SQE_sequential']
            assert self.ml_calc in implemented_kernels, error_not_ml_calc()
            self.kernel_mode = self.ml_calc
            self.kdict = {'k1': {'type': 'gaussian', 'width': 0.4,
                         'dimension': 'features',
                         'bounds': ((1e-3, 0.5),) * len(self.ind_mask_constr),
                         'scaling': 1.0,
                         'scaling_bounds': ((1.0, 1.0), )}
                  }
            self.ml_calc = GPCalculator(
                    kernel_dict=self.kdict, opt_hyperparam=True,
                    scale_data=False,
                    scale_optimizer=False, calc_uncertainty=True,
                    algo_opt_hyperparamters='L-BFGS-B',
                    global_opt_hyperparameters=False,
                    regularization=1e-6, regularization_bounds=(1e-6, 1e-3))
        assert self.ml_calc, error_not_ml_calc()

    def run(self, fmax=0.05, ml_algo='L-BFGS-B', max_iter=500,
            min_iter=0, ml_max_iter=250, penalty=0.0):

        """Executing run will start the optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom).
        ml_algo : string
            Algorithm for the surrogate model. Implemented are:
            'BFGS', 'LBFGS', 'SciPyFminCG', 'SciPyFminBFGS', 'MDMin' and
            'FIRE' as implemented in ASE.
            See https://wiki.fysik.dtu.dk/ase/ase/optimize.html
        max_iter : int
            Max. number of optimization steps.
        min_iter : int
            Min. number of optimizations steps
        ml_max_iter : int
            Max. number of iterations of the machine learning surrogate model.
        penalty : float
            Number of times the predicted energy is penalized w.r.t the
            uncertainty during the machine learning optimization.

        Returns
        -------
        Optimized structure.

        Files:
            1) filename_catlearn.traj contains the trajectory file in ASE
            format of the structures evaluated.
            2) filename_convergence_catlearn.txt contains a summary of the
            optimization process.
            3) An additional file 'warning_and_errors.txt' is included.
        """
        self.ml_algo = ml_algo
        self.list_minimizers_grad = ['BFGS', 'LBFGS', 'SciPyFminCG', 'MDMin',
                                     'FIRE', 'BFGSLineSearch',
                                     'QuasiNewton', 'GoodOldQuasiNewton']

        self.fmax = fmax
        self.min_iter = min_iter
        max_memory = 50

        # Initialization (evaluate two points).
        initialize(self)
        converged(self)
        print_info(self)

        if ml_algo is 'BFGSLineSearch':
            initialize(self, i_step='BFGS')
        if ml_algo not in self.list_minimizers_grad:
            initialize(self, i_step='SciPyFminCG')
        else:
            initialize(self, i_step=ml_algo)

        converged(self)
        print_info(self)

        ase_minimizers = ['BFGS', 'LBFGS', 'SciPyFminCG', 'MDMin',
                        'FIRE', 'BFGSLineSearch', 'SciPyFminBFGS',
                        'QuasiNewton', 'GoodOldQuasiNewton']
        if ml_algo in ase_minimizers:
            self.ase_opt = True
        if ml_algo not in ase_minimizers:
            self.ase_opt = False

        # Configure ML calculator.

        while not converged(self):
            predefined_calculators(self)
            update_prior(self)

            # Limited memory.
            if len(self.list_train) >= max_memory:
                self.list_train = self.list_train[-max_memory:]
                self.list_targets = self.list_targets[-max_memory:]
                self.list_gradients = self.list_gradients[-max_memory:]

            # 1. Train Machine Learning process.

            # Check that the user is not feeding redundant information to ML.
            count_unique = np.unique(self.list_train, return_counts=True,
                                     axis=0)[1]
            msg = 'Your training list contains 1 or more duplicated elements'
            assert np.any(count_unique) < 2, msg
            print('Training a ML process...')
            print('Number of training points:', len(self.list_targets))
            process = train_ml_process(list_train=self.list_train,
                                       list_targets=self.list_targets,
                                       list_gradients=self.list_gradients,
                                       index_constraints=self.ind_mask_constr,
                                       ml_calculator=self.ml_calc,
                                       scaling_targets=self.prior)
            trained_process = process['trained_process']
            ml_calc = process['ml_calc']
            print('ML process trained.')

            # Copy hyperparameters:
            if self.ml_calc.__dict__['opt_hyperparam'] is True:
                self.ml_calc.update_hyperparameters(
                                               trained_process=trained_process)

            # 2. Setup and run optimization.

            # Attach CatLearn calculator.

            # Start from the most stable.
            if self.ase_opt:
                guess = self.ase_ini

                guess_pos = self.list_train[np.argmin(self.list_targets)]
                guess.positions = guess_pos.reshape(-1, 3)

                guess.set_calculator(CatLearnASE(
                                        trained_process=trained_process,
                                        ml_calc=ml_calc,
                                        kappa=penalty,
                                        index_constraints=self.ind_mask_constr
                                             ))
                guess.info['iteration'] = self.iter

                # Run optimization of the predicted PES.
                opt_ml = eval(ml_algo)(guess)
                print('Starting ML optimization...')
                if ml_algo in self.list_minimizers_grad:
                    opt_ml.run(fmax=1e-4, steps=ml_max_iter)
                else:
                    opt_ml.run(steps=ml_max_iter)
                print('ML optimized.')

                interesting_point = guess.get_positions().flatten()

            if not self.ase_opt:
                x0 = self.list_train[np.argmin(self.list_targets)]

                x0 = np.array(apply_mask_ase_constraints(list_to_mask=[x0],
                mask_index=self.ind_mask_constr)[1])

                int_p = optimize_ml_using_scipy(x0=x0,
                                        ml_calc=self.ml_calc,
                                        trained_process=trained_process,
                                        ml_algo=ml_algo)
                interesting_point = unmask_geometry(
                                    org_list=self.list_train,
                                    masked_geom=int_p,
                                    mask_index=self.ind_mask_constr)

            # 3. Evaluate and append interesting point.
            eval_and_append(self, interesting_point)

            # 4. Convergence and output.

            # Save evaluated image.
            TrajectoryWriter(atoms=self.ase_ini,
                             filename='./' + str(self.filename),
                             mode='a').write()

            # Printing:
            self.list_fmax = get_fmax(-np.array([self.list_gradients[-1]]),
                                  self.num_atoms)
            self.max_abs_forces = np.max(np.abs(self.list_fmax))
            print_info(self)
            # Maximum number of iterations reached.
            if self.iter > max_iter:
                print('Not converged. Maximum number of iterations reached.')
                break


def initialize(self, i_step=1e-3):
    """ The GPR model needs two points to start the ML surrogate model.
    This function takes care of the two first optimization steps.
    First, evaluates the "real" function for a given initial guess. Secondly,
    obtains the function value of a second guessed point originated using the
    same optimizor used for the optimization of the predicted PES. This
    function is exclusively called when the optimization is initialized and
    the user has not provided any trained data.
    """

    if len(self.list_targets) == 1:

        if self.jac is False:
            alpha = np.random.normal(loc=0.0, scale=i_step,
                                     size=np.shape(self.list_train[0]))
            ini_train = [self.list_train[-1] - alpha]

        if isinstance(i_step, float):
            if self.jac is True:
                alpha = i_step + np.zeros_like(self.list_train[0])
                ini_train = [self.list_train[-1] - alpha *
                             self.list_gradients[-1]]

        if isinstance(i_step, str):
            if i_step in self.list_minimizers_grad:
                eval(i_step)(self.ase_ini).run(fmax=0.05, steps=1)
            else:
                eval(i_step)(self.ase_ini).run(steps=1)
            ini_train = [self.ase_ini.get_positions().flatten()]

        self.list_train = np.append(self.list_train, ini_train, axis=0)
        self.list_targets = np.append(self.list_targets,
                                      get_energy_catlearn(self))
        self.feval += 1
        self.list_targets = np.reshape(self.list_targets,
                                       (len(self.list_targets), 1))
        if self.jac is True:
            self.list_gradients = np.append(
                                        self.list_gradients,
                                        -get_forces_catlearn(self).flatten())
            self.list_gradients = np.reshape(
                                        self.list_gradients,
                                        (len(self.list_targets), np.shape(
                                            self.list_train)[1])
                                        )
        self.feval = len(self.list_targets)

        if self.ase:
            molec_writer = TrajectoryWriter('./' + str(self.filename),
                                            mode='a')
            molec_writer.write(self.ase_ini)
        self.iter += 1

    if len(self.list_targets) == 0:
        self.list_targets = [np.append(self.list_targets,
                             get_energy_catlearn(self))]
        self.feval += 1
        if self.jac is True:
            self.list_gradients = [np.append(self.list_gradients,
                                   -get_forces_catlearn(self).flatten())]
        self.feval = len(self.list_targets)
        if self.ase:
            molec_writer = TrajectoryWriter('./' + str(self.filename),
                                            mode='w')
            molec_writer.write(self.ase_ini)


def update_prior(self):
    self.prior = np.max(self.list_targets)
    if self.max_abs_forces <= 0.10:
        self.prior = np.min(self.list_targets) + 2.0


def predefined_calculators(self):
    """ List of predefined kernels setups."""

    if self.kernel_mode is 'SQE_static':
        print('Using static SQE kernel. Hyperparameters are kept fixed.')
        self.ml_calc.__dict__['opt_hyperparam'] = False

    if self.kernel_mode is 'SQE_isotropic':
        print('Using default isotropic SQE kernel. A common length-scale '
              'paramter is optimized for all dimensions.')
        self.ml_calc.__dict__['regularization_bounds'] = (1e-6, 1e-3)
        self.ml_calc.__dict__['kdict']['k1']['dimension'] = 'single'
        self.ml_calc.__dict__['kdict']['k1']['width'] = 0.01
        self.ml_calc.__dict__['kdict']['k1']['bounds'] = ((1e-4, 0.40),)

    if self.kernel_mode is 'SQE_anisotropic':
        print('Using default anisotropic ARD-SQE kernel. Length-scale '
              'hyperparameters are optimized for each dimension.')

    if self.kernel_mode is 'SQE_sequential':

        # Step 1:
        if self.max_abs_forces > 2.00:
            print('Sequential mode. Stage 1: ARD-SQE (anisotropic).')
            self.ml_calc.__dict__['regularization_bounds'] = (1e-5, 1e-2)
            self.ml_calc.__dict__['kdict']['k1']['dimension'] = 'features'
            self.ml_calc.__dict__['kdict']['k1']['width'] = 0.25
            self.ml_calc.__dict__['kdict']['k1']['bounds'] = ((1e-4, 1.0),
            ) * len(self.ind_mask_constr)

        # Step 2:
        if 0.05 < self.max_abs_forces <= 2.00:
            print('Sequential mode. Stage 2: SQE (isotropic).')
            self.ml_calc.__dict__['regularization_bounds'] = (1e-8, 1e-3)
            self.ml_calc.__dict__['kdict']['k1']['dimension'] = 'single'
            self.ml_calc.__dict__['kdict']['k1']['width'] = 0.1
            self.ml_calc.__dict__['kdict']['k1']['bounds'] = ((1e-4, 0.4),)
        # Step 3:
        if self.max_abs_forces <= 0.05:
            print('Sequential mode. Stage 3: SQE (fixed hyperparameters).')
            self.ml_calc.__dict__['opt_hyperparam'] = False
            update_bound = 0.4
            self.ml_calc.__dict__['regularization'] = 1e-10
            self.ml_calc.__dict__['regularization_bounds'] = (1e-8, 1e-3)
            self.ml_calc.__dict__['kdict']['k1']['dimension'] = 'single'
            self.ml_calc.__dict__['kdict']['k1']['width'] = update_bound
            self.ml_calc.__dict__['kdict']['k1']['bounds'] = ((1e-4,
                                                               update_bound),)