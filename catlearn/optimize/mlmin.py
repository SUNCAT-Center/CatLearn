import os.path
import numpy as np
from scipy.spatial.distance import euclidean

from ase.optimize import MDMin
from ase.io import read, write
from ase.io.trajectory import TrajectoryWriter
from ase.parallel import parprint, parallel_function

from catlearn.optimize.io import print_info, print_version, print_cite_mlmin
from catlearn.optimize.constraints import create_mask, apply_mask
from catlearn.optimize.mlneb import ASECalc
from catlearn.regression import GaussianProcess
from catlearn import __version__
from catlearn.active_learning.acquisition_functions import UCB


class MLMin(object):

    def __init__(self, x0, prev_calculations=None, restart=False,
                 trajectory='catlearn_opt.traj', force_consistent=None):

        """Optimization setup.

        Parameters
        ----------
        x0 : Atoms object or trajectory file in ASE format.
            Initial guess.
        restart: boolean
            Restart calculation. Only if a trajectory file with the same
            name as the one introduced in the 'trajectory' flag is found.
            This file must be in the same directory than the direction in
            which the optimization was initiated.
        prev_calculations: list of Atoms or trajectory file (ASE format).
            The user can feed previously calculated data for the same
            hypersurface (i.e. the calculations must be done using the same
            calculator and parameters).
        trajectory: string
            Filename to store the output.
        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.

        """

        # General variables.
        self.prev_calculations = prev_calculations
        self.filename = trajectory
        self.iter = 0
        self.fmax = 0.
        self.gp = None
        self.opt_type = 'MLMin'
        self.version = self.opt_type + ' ' + __version__
        self.fc = force_consistent
        print_version(self.version)

        if restart is True and prev_calculations is None:
            if os.path.isfile(self.filename):
                self.prev_calculations = self.filename

        msg = 'You must set an initial Atoms structure.'
        assert x0 is not None, msg

        # Read trajectory file.
        if isinstance(x0, str):
            x0 = read(x0, '-1')

        self.ase_ini = x0
        self.ase_calc = self.ase_ini.get_calculator()
        msg = 'ASE calculator not found.'
        assert self.ase_calc, msg
        self.constraints = self.ase_ini.constraints
        self.x0 = self.ase_ini.get_positions().flatten()
        self.num_atoms = self.ase_ini.get_number_of_atoms()

        # Information from the initial structure.
        new_atoms = self.ase_ini
        self.list_atoms = [new_atoms]

        if self.prev_calculations is not None:
            if isinstance(self.prev_calculations, str):
                parprint('Reading previous calculations from a traj. file.')
                self.prev_calculations = read(self.prev_calculations, ':')

            # Check for duplicates.
            for prev_atoms in self.prev_calculations:
                duplicate = False
                prev_pos = prev_atoms.get_positions().reshape(-1)
                for atoms in self.list_atoms:
                    pos_atoms = atoms.get_positions().reshape(-1)
                    d_ij = euclidean(pos_atoms, prev_pos)
                    if d_ij == 0:
                        duplicate = True
                if duplicate is False:
                    self.list_atoms += [prev_atoms]

        # Extract information from previous calculations.
        self.list_train = []
        self.list_targets = []
        self.list_gradients = []
        for i in range(0, len(self.list_atoms)):
            pos_atoms = list(self.list_atoms[i].get_positions().reshape(-1))
            energy_atoms = self.list_atoms[i].get_potential_energy(
                                                      force_consistent=self.fc)
            grad_atoms = list(-self.list_atoms[i].get_forces().reshape(-1))
            self.list_train.append(pos_atoms)
            self.list_targets.append(energy_atoms)
            self.list_gradients.append(grad_atoms)
        self.list_max_abs_forces = []
        for i in self.list_gradients:
            self.list_fmax = get_fmax(np.array([i]))
            self.max_abs_forces = np.max(np.abs(self.list_fmax))
            self.list_max_abs_forces.append(self.max_abs_forces)

        # Get constraints.
        self.index_mask = create_mask(self.ase_ini, self.constraints)

        # Dump all Atoms to file.
        write(self.filename, self.list_atoms)
        print_info(self)

    def run(self, fmax=0.05, steps=200, kernel='SQE', max_step=0.25,
            acq='min_energy', full_output=False, noise=0.005):

        """Executing run will start the optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angstrom).
        steps : int
            Max. number of optimization steps.
        kernel: string
            Type of covariance function to be used.
            Implemented are: SQE (fixed hyperparamters), SQE_opt and ARD_SQE.
        max_step: float
            Early stopping criteria. Maximum uncertainty before stopping the
            optimization in the predicted landscape.
        acq : string
            The acquisition function that decides the next point to
            evaluate. Implemented are: 'lcb', 'ucb', 'min_energy'.
        full_output: boolean
            Whether to print on screen the full output (True) or not (False).
        noise:
            Regularization parameter of the GP.
        Returns
        -------
        Optimized atom structure.

        """

        self.fmax = fmax
        self.acq = acq
        success_hyper = False

        while not converged(self):

            # 1. Train Machine Learning model.
            train = np.copy(self.list_train)
            targets = np.copy(self.list_targets)
            gradients = np.copy(self.list_gradients)

            self.u_prior = np.max(targets)
            scaled_targets = targets - self.u_prior
            sigma_f = 1e-3 + np.std(scaled_targets) ** 2

            if kernel == 'SQE_fixed':
                opt_hyper = False
                kdict = [{'type': 'gaussian', 'width': 0.4,
                          'dimension': 'single',
                          'bounds': ((0.4, 0.4),),
                          'scaling': 1.0,
                          'scaling_bounds': ((1.0, 1.0),)},
                         {'type': 'noise_multi',
                          'hyperparameters': [noise * 0.4**2, noise],
                          'bounds': ((noise * 0.4**2, noise * 0.4**2),
                                     (noise, noise),)}
                         ]

            if kernel == 'SQE':
                opt_hyper = True
                kdict = [{'type': 'gaussian', 'width': 0.4,
                          'dimension': 'single',
                          'bounds': ((0.01, 1.0),),
                          'scaling': sigma_f,
                          'scaling_bounds': ((sigma_f, sigma_f),)},
                         {'type': 'noise_multi',
                          'hyperparameters': [noise, noise * 0.4**2],
                          'bounds': ((noise/5., noise*10.),
                                     (noise/5. * 0.4**2, noise*10.),)}
                         ]

            if kernel == 'ARD_SQE':
                opt_hyper = True
                kdict = [{'type': 'gaussian', 'width': 0.4,
                          'dimension': 'features',
                          'bounds': ((0.01, 1.0),) * len(self.index_mask),
                          'scaling': sigma_f,
                          'scaling_bounds': ((sigma_f, sigma_f),)},
                         {'type': 'noise_multi',
                          'hyperparameters': [noise/2., noise/10.],
                          'bounds': ((noise/5., noise),
                                     (noise/10., noise/4.),)}
                         ]

            if len(self.list_targets) == 1:
                opt_hyper = False
                kdict = [{'type': 'gaussian', 'width': 0.20,
                          'dimension': 'single',
                          'bounds': ((0.20, 0.20),),
                          'scaling': sigma_f,
                          'scaling_bounds': ((sigma_f, sigma_f),)},
                         {'type': 'noise_multi',
                          'hyperparameters': [noise * 0.4**2, noise],
                          'bounds': ((noise * 0.4**2, noise * 0.4**2),
                                     (noise, noise),)}
                         ]

            if success_hyper is not False:
                kdict = success_hyper

            if self.index_mask is not None:
                train = apply_mask(list_to_mask=train,
                                   mask_index=self.index_mask)[1]
                gradients = apply_mask(list_to_mask=gradients,
                                       mask_index=self.index_mask)[1]

            parprint('Training a GP process...')
            parprint('Number of training points:', len(scaled_targets))

            train = train.tolist()
            gradients = gradients.tolist()

            self.gp = fit(train, scaled_targets, gradients, kdict)

            if opt_hyper is True:
                if len(self.list_targets) > 5:
                    self.gp.optimize_hyperparameters()
                    if self.gp.theta_opt.success is True:
                        if full_output is True:
                            parprint('Hyperparam. optimization was successful.')
                            parprint('Updating kernel list...')
                        success_hyper = self.gp.kernel_list

                    if self.gp.theta_opt.success is not True:
                        if full_output is True:
                            parprint('Hyperparam. optimization unsuccessful.')
                        if success_hyper is False:
                            if full_output is True:
                                parprint('Not enough data...')
                        if success_hyper is not False:
                            if full_output is True:
                                parprint('Using the last optimized '
                                      'hyperparamters.')
                    if full_output is True:
                        parprint('Kernel list:', self.gp.kernel_list)

            # 2. Optimize Machine Learning model.

            self.list_interesting_points = []
            self.list_interesting_energies = []
            self.list_interesting_uncertainties = []

            guess = self.ase_ini
            guess_pos = np.array(self.list_train[-1])
            guess.positions = guess_pos.reshape(-1, 3)
            guess.set_calculator(ASECalc(gp=self.gp,
                                         index_constraints=self.index_mask,
                                         scaling_targets=self.u_prior)
                                 )

            # Optimization in the predicted landscape:
            ml_opt = MDMin(guess, trajectory=None, logfile=None, dt=0.020)

            if full_output is True:
                parprint('Starting optimization on the predicted landscape...')
            ml_converged = False

            n_steps_performed = 0

            while ml_converged is False:
                ml_opt.run(fmax=fmax*0.90, steps=1)
                pos_ml = np.array(guess.positions).flatten()
                self.list_interesting_points.append(pos_ml)
                pos_ml = apply_mask([pos_ml],
                                    mask_index=self.index_mask)[1]
                pred_ml = self.gp.predict(test_fp=pos_ml, uncertainty=True)
                energy_ml = pred_ml['prediction'][0][0]
                unc_ml = pred_ml['uncertainty_with_reg'][0]

                self.list_interesting_energies.append(energy_ml)
                self.list_interesting_uncertainties.append(unc_ml)

                n_steps_performed += 1
                if n_steps_performed > 1000:
                    if full_output is True:
                        parprint('Not converged yet...')
                        ml_converged = True
                if unc_ml >= max_step:
                    if full_output is True:
                        parprint('Maximum uncertainty reach. Early stop.')
                    ml_converged = True
                if ml_opt.converged():
                    if full_output is True:
                        parprint('ML optimized.')
                    ml_converged = True

            # Acquisition functions:
            acq_pred = np.array(self.list_interesting_energies)
            acq_unc = np.array(self.list_interesting_uncertainties)

            if self.acq == 'ucb':
                acq_values = UCB(predictions=acq_pred, uncertainty=acq_unc,
                                 objective='min', kappa=-1.0)
            if self.acq == 'lcb':
                e_minus_unc = np.array(self.list_interesting_energies) - \
                              np.array(self.list_interesting_uncertainties)
                acq_values = -e_minus_unc
            if self.acq == 'min_energy':
                acq_values = -np.array(self.list_interesting_energies)

            max_score = np.argmax(acq_values)

            self.interesting_point = self.list_interesting_points[max_score]

            # 3. Evaluate and append interesting point.
            if full_output is True:
                parprint('Performing evaluation in the real landscape...')

            eval_atom = self.ase_ini
            pos_atom = self.interesting_point
            eval_atom.positions = np.array(pos_atom).reshape((-1, 3))
            eval_atom.set_calculator(self.ase_calc)
            energy_atom = eval_atom.get_potential_energy(
                                                      force_consistent=self.fc)
            forces_atom = -eval_atom.get_forces().reshape(-1)

            # 4. Convergence and output.
            self.list_train.append(pos_atom)
            self.list_targets.append(energy_atom)
            self.list_gradients.append(forces_atom)

            self.list_fmax = get_fmax(np.array([self.list_gradients[-1]]))
            self.max_abs_forces = np.max(np.abs(self.list_fmax))
            self.list_max_abs_forces.append(self.max_abs_forces)

            self.iter += 1

            print_info(self)

            # Save evaluated image.
            self.list_atoms += [eval_atom]
            TrajectoryWriter(atoms=self.ase_ini,
                             filename=self.filename,
                             mode='a').write()

            # Maximum number of iterations reached.
            if self.iter >= steps:
                parprint('Not converged. Maximum number of iterations reached.')
                break


@parallel_function
def fit(X, y, gradients, kdict):
    """ Train the Gaussian process."""
    gp = GaussianProcess(kernel_list=kdict,
                         regularization=0.0,
                         regularization_bounds=(0.0, 0.0),
                         train_fp=X,
                         train_target=y,
                         gradients=gradients,
                         optimize_hyperparameters=False)
    print('Gaussiaon Process process trained.')
    
    return gp


def converged(self):
    """Function that checks the convergence of the min. surrogate model."""

    self.list_fmax = get_fmax(np.array([self.list_gradients[-1]]))
    self.max_abs_forces = np.max(np.abs(self.list_fmax))

    if self.max_abs_forces < self.fmax:
        parprint('Congratulations. Structural optimization has converged.')
        parprint('All the evaluated structures can be found in:',
              self.filename)
        print_cite_mlmin()
        return True
    return False


def get_fmax(gradients_flatten):
    """Function that print a list of max. individual atom forces."""
    forces_flatten = -gradients_flatten
    list_fmax = np.zeros((len(gradients_flatten), 1))
    j = 0
    for i in forces_flatten:
        atoms_forces_i = np.reshape(i, (-1, 3))
        list_fmax[j] = np.max(np.sqrt(np.sum(atoms_forces_i**2, axis=1)))
        j = j + 1
    return list_fmax
