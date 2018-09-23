# CatLearn 1.5.0

from ase import Atoms
from ase.io.trajectory import TrajectoryWriter
from ase.optimize import *
from ase.optimize.sciopt import *
from catlearn.optimize.warnings import *
from catlearn.optimize.io import ase_traj_to_catlearn, print_info, \
                                 print_version
from catlearn.optimize.constraints import create_mask, unmask_geometry, \
                                          apply_mask
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
        self.version = 'Min. v.1.6.0'
        print_version(self.version)

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
            self.list_train = np.array([])
            self.list_targets = np.array([])
            self.list_gradients = np.array([])
            if len(self.constraints) < 0:
                self.constraints = None
            if self.constraints is not None:
                self.index_mask = create_mask(self.ase_ini, self.constraints)

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
                self.index_mask = create_mask(self.ase_ini, self.constraints)
            self.feval = len(self.list_targets)
        self.d_i_i = 0.0

    def run(self, fmax=0.05, ml_algo='L-BFGS-B', steps=200, alpha=1e-2,
            min_iter=0, ml_max_steps=250, max_memory=50):

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
        ml_max_steps : int
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
        self.max_memory = max_memory

        # Initialization (evaluate two points).

        if len(self.list_targets) == 0:
            self.list_train = [self.ase_ini.get_positions().flatten()]
            self.list_targets = [np.append(self.list_targets,
                                 get_energy_catlearn(self))]
            self.list_gradients = [np.append(self.list_gradients,
                                   -get_forces_catlearn(self).flatten())]
            self.feval += 1
            molec_writer = TrajectoryWriter('./' + str(self.filename),
                                            mode='w')
            molec_writer.write(self.ase_ini)

        converged(self)
        print_info(self)

        if not converged(self):
            i_step = alpha + np.zeros_like(self.list_train[0])
            steepest_geometry = [self.list_train[0] - i_step *
                                 self.list_gradients[0]]
            eval_and_append(self, steepest_geometry)
            molec_writer = TrajectoryWriter('./' + str(self.filename),
                                            mode='a')
            molec_writer.write(self.ase_ini)
        converged(self)
        print_info(self)

        ase_minimizers = ['BFGS', 'LBFGS', 'SciPyFminCG', 'MDMin',
                          'FIRE', 'BFGSLineSearch', 'SciPyFminBFGS',
                          'QuasiNewton', 'GoodOldQuasiNewton']

        if ml_algo in ase_minimizers:
            self.ase_opt = True
        if ml_algo not in ase_minimizers:
            self.ase_opt = False

        while not converged(self):

            # 1. Train Machine Learning model.
            train_gp_model(self)

            # 2. Optimize Machine Learning model.

            interesting_point = []

            if self.ase_opt is True:
                guess = self.ase_ini
                # guess_pos = self.list_train[np.argmin(self.list_targets)]
                guess_pos = self.list_train[0]
                guess.positions = guess_pos.reshape(-1, 3)
                guess.set_calculator(CatLearnASE(
                                        gp=self.gp,
                                        index_constraints=self.index_mask,
                                        scaling_targets=self.max_target)
                                     )
                guess.info['iteration'] = self.iter

                # Run optimization of the predicted PES.
                opt_ml = eval(ml_algo)(guess, logfile=None)
                print('Starting ML optimization...')
                opt_ml.run(fmax=1e-4, steps=ml_max_steps)
                print('ML optimized.')

                interesting_point = guess.get_positions().flatten()

            if self.ase_opt is False:
                x0 = self.list_train[np.argmin(self.list_targets)]
                x0 = np.array(apply_mask(list_to_mask=[x0],
                              mask_index=self.index_mask)[1])

                int_p = optimize_ml_using_scipy(x0=x0, gp=self.gp,
                                                scaling=self.max_target,
                                                ml_algo=ml_algo)
                interesting_point = unmask_geometry(
                                    org_list=self.list_train,
                                    masked_geom=int_p,
                                    mask_index=self.index_mask)

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


def train_gp_model(self):
    self.max_target = np.abs(np.max(self.list_targets))
    # self.max_target = np.abs(np.min(self.list_targets)) + 2.0 * np.abs(
    # self.list_targets[-1]-self.list_targets[0])
    scaled_targets = self.list_targets.copy() - self.max_target

    # from scipy.spatial.distance import euclidean
    # self.d_i_i += euclidean(self.list_train[0], self.list_train[-1])

    width = 1.0
    dimension = 'single'
    bounds = ((0.1, width),)

    kdict = [{'type': 'gaussian', 'width': width,
              'dimension': dimension,
              'bounds': bounds,
              'scaling': 1.,
              'scaling_bounds': ((1., 1.),)},
             ]

    train = self.list_train.copy()
    gradients = self.list_gradients.copy()
    if self.index_mask is not None:
        train = apply_mask(list_to_mask=self.list_train,
                           mask_index=self.index_mask)[1]
        gradients = apply_mask(list_to_mask=self.list_gradients,
                               mask_index=self.index_mask)[1]

    if len(train) >= self.max_memory:
        train = train[-self.max_memory:]
        scaled_targets = scaled_targets[-self.max_memory:]
        gradients = gradients[-self.max_memory:]


    print('Training a GP process...')
    print('Number of training points:', len(scaled_targets))

    self.gp = GaussianProcess(kernel_dict=kdict,
                         regularization=0.001,
                         regularization_bounds=(0.0001, 0.001),
                         train_fp=train,
                         train_target=scaled_targets,
                         gradients=gradients,
                         optimize_hyperparameters=False,
                         scale_data=False)
    self.gp.optimize_hyperparameters(global_opt=False)
    print('Optimized hyperparameters:', self.gp.theta_opt)
    print('GP process trained.')

