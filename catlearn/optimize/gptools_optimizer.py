# CatLearn + GPTools.

from ase import Atoms
from ase.io.trajectory import TrajectoryWriter
from catlearn.optimize.warnings import *
from catlearn.optimize.io import ase_traj_to_catlearn, print_info, \
                                 print_version
from catlearn.optimize.constraints import create_mask, unmask_geometry, \
                                          apply_mask
from catlearn.optimize.get_real_values import eval_and_append, \
                                              get_energy_catlearn, \
                                              get_forces_catlearn
from catlearn.optimize.convergence import converged, get_fmax
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import gptools


class GPTMin(object):

    def __init__(self, x0, ase_calc=None, trajectory='catlearn_opt.traj'):

        """Optimization setup.

        Parameters
        ----------
        x0 : Atoms object or trajectory file in ASE format.
            Initial guess.
        ase_calc: ASE calculator object.
            When using ASE the user must pass an ASE calculator.
        trajectory: string
            Filename to store the output.
        """

        # General variables.
        self.filename = trajectory  # Remove extension if added.
        self.iter = 0
        self.feval = 0
        self.fmax = 0.0
        self.min_iter = 0
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
            self.list_max_abs_forces = []

        if isinstance(x0, str):
            self.start_mode = 'trajectory'
            self.ase = True
            self.ase_calc = ase_calc
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
            self.list_max_abs_forces = []
            for i in self.list_gradients:
                self.list_fmax = get_fmax(-np.array([i]), self.num_atoms)
                self.max_abs_forces = np.max(np.abs(self.list_fmax))
                self.list_max_abs_forces.append(self.max_abs_forces)

    def run(self, fmax=0.05, steps=200, min_iter=0,
            kernel='RationalQuadratic', optimize_hyperparameters=False):

        """Executing run will start the optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom).
        steps : int
            Max. number of optimization steps.
        min_iter : int
            Min. number of optimizations steps

        Returns
        -------
        Optimized atom structure.

        """

        self.fmax = fmax
        self.min_iter = min_iter

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

        self.list_max_abs_forces.append(self.max_abs_forces)
        print_info(self)

        while not converged(self):

            # 1. Train Machine Learning model.
            train = self.list_train
            targets = self.list_targets
            gradients = self.list_gradients

            u_prior = np.max(targets)

            scaled_targets = targets - u_prior

            n_dim = len(self.index_mask)

            if self.index_mask is not None:
                train = apply_mask(list_to_mask=train,
                                   mask_index=self.index_mask)[1]
                gradients = apply_mask(list_to_mask=gradients,
                                       mask_index=self.index_mask)[1]

            print('Training a GP process...')
            print('Number of training points:', len(scaled_targets))

            if kernel == 'RationalQuadratic':
                gp_bounds = [(1., 1.)] + [(1.0, 1.0)] + [(0.4, 0.4)] * \
                             n_dim

                kernel = gptools.RationalQuadraticKernel(
                                               param_bounds=gp_bounds,
                                               num_dim=n_dim)

                self.gp = gptools.GaussianProcess(kernel)

                self.gp.free_params = np.ones(n_dim + 2) * 0.4  # l1, l2, l3
                self.gp.free_params[0] = 1.0  # sigma_f
                self.gp.free_params[1] = 1.0  # alpha
                print('Hyperparameters:', self.gp.free_param_names)

            if kernel == 'Matern52':
                gp_bounds = [(1., 1.)] + [(0.4, 1.0)] * n_dim
                kernel = gptools.Matern52Kernel(
                                               param_bounds=gp_bounds,
                                               num_dim=n_dim)
                self.gp = gptools.GaussianProcess(kernel)
                self.gp.free_params = np.ones(n_dim + 1) * 0.4  # l1, l2, l3
                self.gp.free_params[0] = 1.0  # sigma_f

                print('Hyperparamers:', self.gp.free_param_names)

            if kernel == 'SQE':
                n_dim = len(self.index_mask)
                gp_bounds = [(1., 1.)] + [(0.4, 0.4)] * n_dim
                kernel = gptools.SquaredExponentialKernel(
                                               param_bounds=gp_bounds,
                                               num_dim=n_dim)
                self.gp = gptools.GaussianProcess(kernel)
                self.gp.free_params = np.ones(n_dim + 1) * 0.4  # l1, l2, l3
                self.gp.free_params[0] = 1.0  # sigma_f
                print('Hyperparameters:', self.gp.free_param_names)

            self.gp.add_data(train, scaled_targets.flatten(), err_y=0.005)

            for i in range(0, np.shape(gradients)[1]):
                g_i = gradients[:, i]
                n_i = np.zeros(np.shape(gradients))
                n_i[:, i] = 1.0
                self.gp.add_data(train, g_i, n=n_i, err_y=0.005*0.4**2)

            if optimize_hyperparameters is True:
                try:
                    self.gp.optimize_hyperparameters()
                except:
                    pass

            print('GP process trained.')

            # 2. Optimize Machine Learning model.

            def predicted_energy_test(x0, gp, u_prior=0.0):
                return gp.predict(x0, return_std=False)[0] + u_prior

            guess = self.list_train[-1]
            guess = np.array(apply_mask(list_to_mask=[guess],
                                        mask_index=self.index_mask)[1])

            args = (self.gp, u_prior,)

            result_min = fmin_l_bfgs_b(func=predicted_energy_test,
                                       approx_grad=True,
                                       x0=guess,
                                       args=args)
            pred_pos = result_min[0]

            interesting_point = unmask_geometry(
                                    org_list=self.list_train,
                                    masked_geom=pred_pos,
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
