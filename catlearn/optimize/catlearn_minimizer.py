# CatLearn 1.0.5

from ase import Atoms
from ase.io.trajectory import TrajectoryWriter
from ase.optimize import *
from ase.optimize.sciopt import *
from catlearn.optimize.warnings import *
from catlearn.optimize.io import ase_traj_to_catlearn, print_info
from catlearn.optimize.constraints import create_mask_ase_constraints, \
                                    unmask_geometry, apply_mask_ase_constraints
from catlearn.optimize.get_real_values import eval_and_append, \
                                              get_energy_catlearn, \
                                              get_forces_catlearn
from catlearn.optimize.convergence import converged, get_fmax
from catlearn.optimize.catlearn_ase_calc import CatLearnASE, \
                                                optimize_ml_using_scipy
import numpy as np
from catlearn.regression import GaussianProcess


class CatLearnMin(object):

    def __init__(self, x0, ase_calc=None, ml_calc='SQE',
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
        self.version = 'Min. v.1.0.7'

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

    def run(self, fmax=0.05, ml_algo='LBFGS', steps=200,
            min_iter=0, ml_max_iter=250, max_memory=50):

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
        steps : int
            Max. number of optimization steps.
        min_iter : int
            Min. number of optimizations steps
        ml_max_iter : int
            Max. number of iterations of the machine learning surrogate model.

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
        # Default kernel:

        self.ml_algo = ml_algo
        self.fmax = fmax
        self.min_iter = min_iter

        # Initialization (evaluate two points).
        initialize(self)
        converged(self)
        print_info(self)

        initialize(self, i_step='BFGS')

        converged(self)
        print_info(self)

        ase_minimizers = ['BFGS', 'LBFGS', 'SciPyFminCG',
                          'FIRE', 'BFGSLineSearch', 'SciPyFminBFGS',
                          'QuasiNewton', 'GoodOldQuasiNewton']

        if ml_algo in ase_minimizers:
            self.ase_opt = True
        if ml_algo not in ase_minimizers:
            self.ase_opt = False

        while not converged(self):

            # Configure ML calculator.

            max_target = np.max(self.list_targets)
            scaled_targets = self.list_targets.copy() - max_target
            scaling = 0.1 + np.std(scaled_targets)**2

            width = 0.4
            noise_energy = 0.0005
            noise_forces = 0.0005 * width**2

            kdict = [{'type': 'gaussian', 'width': width,
                            'dimension': 'single',
                            'bounds': ((width, width),),
                            'scaling': scaling,
                            'scaling_bounds': ((scaling, scaling+100.0),)},
                     {'type': 'noise_multi',
                            'hyperparameters': [noise_energy, noise_forces],
                            'bounds': ((noise_energy, 1e-2),
                                       (noise_forces, 1e-2),)}
                     ]

            # 1. Train Machine Learning process.
            train = self.list_train.copy()
            gradients = self.list_gradients.copy()

            if self.ind_mask_constr is not None:
                train = apply_mask_ase_constraints(
                                   list_to_mask=self.list_train,
                                   mask_index=self.ind_mask_constr)[1]
                gradients = apply_mask_ase_constraints(
                                        list_to_mask=self.list_gradients,
                                        mask_index=self.ind_mask_constr)[1]

            # Limited memory.
            if len(train) >= max_memory:
                train = train[-max_memory:]
                scaled_targets = scaled_targets[-max_memory:]
                gradients = gradients[-max_memory:]

            print('Training a GP process...')
            print('Number of training points:', len(scaled_targets))

            gp = GaussianProcess(kernel_dict=kdict,
                                 regularization=0.0,
                                 regularization_bounds=(0.0, 0.0),
                                 train_fp=train,
                                 train_target=scaled_targets,
                                 gradients=gradients,
                                 optimize_hyperparameters=False,
                                 scale_data=False)
            gp.optimize_hyperparameters(global_opt=False)
            print('GP process trained.')

            # 2. Optimize Machine Learning process.

            interesting_point = []

            if self.ase_opt is True:
                guess = self.ase_ini
                guess_pos = self.list_train[np.argmin(self.list_targets)]
                guess.positions = guess_pos.reshape(-1, 3)
                guess.set_calculator(CatLearnASE(
                                        gp=gp,
                                        index_constraints=self.ind_mask_constr)
                                     )
                guess.info['iteration'] = self.iter

                # Run optimization of the predicted PES.
                opt_ml = eval(ml_algo)(guess, logfile=None)
                print('Starting ML optimization...')
                opt_ml.run(fmax=1e-4, steps=ml_max_iter)
                print('ML optimized.')

                interesting_point = guess.get_positions().flatten()

            if self.ase_opt is False:
                x0 = self.list_train[np.argmin(self.list_targets)]
                x0 = np.array(apply_mask_ase_constraints(list_to_mask=[x0],
                              mask_index=self.ind_mask_constr)[1])

                int_p = optimize_ml_using_scipy(x0=x0, gp=gp,
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
            if self.iter >= steps:
                print('Not converged. Maximum number of iterations reached.')
                break


def initialize(self, i_step='BFGS'):
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
            eval(i_step)(self.ase_ini, logfile=None).run(fmax=0.05,
                                                             steps=1)
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
