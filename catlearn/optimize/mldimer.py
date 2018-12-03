from ase import Atoms
from ase.io.trajectory import TrajectoryWriter
from ase.io import Trajectory, read
from ase.calculators.calculator import Calculator, all_changes
from catlearn.optimize.warnings import *
from catlearn.optimize.io import ase_traj_to_catlearn, print_info, \
                                 print_version
from catlearn.optimize.constraints import create_mask, unmask_geometry, \
                                          apply_mask
from catlearn.optimize.get_real_values import eval_and_append, \
                                              get_energy_catlearn, \
                                              get_forces_catlearn
from catlearn.optimize.convergence import converged_dimer, get_fmax
from scipy.optimize import fmin_l_bfgs_b
from ase.constraints import FixAtoms
import numpy as np
from catlearn.regression import GaussianProcess
import copy
from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
import matplotlib.pyplot as plt
from catlearn.optimize.functions_calc import Himmelblau


class MLDimer(object):

    def __init__(self, x0, ase_calc=None, trajectory='mldimer_opt.traj'):

        """ ML-Dimer Method

        Parameters
        ----------
        start: Trajectory file (in ASE format) or Atoms object.
            Initial end-point of the NEB path or Atoms object.
        vector: array or string
            Manual: Vector array for the displacement of the dimer.
            Automatic: 'random' vector
        ase_calc: ASE calculator Object.
            ASE calculator as implemented in ASE.
            See https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
        prev_calculations: Atoms list or Trajectory file (in ASE format).
            (optional) The user can feed previously calculated data for the
            same hypersurface. The previous calculations must be fed as an
            Atoms list or Trajectory file.
        restart: boolean
            Only useful if you want to continue your ML-NEB in the same
            directory. The file "evaluated_structures.traj" from the
            previous run, must be located in the same run directory.
        """

        # General variables.
        self.filename = trajectory  # Remove extension if added.
        self.iter = 0
        self.feval = 0
        self.fmax = 0.0
        self.min_iter = 0
        self.gp = None
        self.version = 'Dimer v.0.1.0'
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
            self.ase_ini = trj_images[0]
            molec_writer = TrajectoryWriter('./' + str(self.filename),
                                            mode='w')
            molec_writer.write(self.ase_ini)
            for i in range(1, len(trj_images)):
                self.ase_ini = trj_images[i]
                molec_writer = TrajectoryWriter('./' + str(self.filename),
                                                mode='a')
                molec_writer.write(self.ase_ini)
            self.feval = len(self.list_targets)
            self.list_max_abs_forces = []
            for i in self.list_gradients:
                self.list_fmax = get_fmax(-np.array([i]), self.num_atoms)
                self.max_abs_forces = np.max(np.abs(self.list_fmax))
                self.list_max_abs_forces.append(self.max_abs_forces)
        self.index_mask = create_mask(self.ase_ini, self.constraints)

    def run(self, dmask, vector, fmax=0.05, steps=200, kernel='SQE',
            max_uncertainty=0.200):
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
        max_uncertainty: float
            Early stopping criteria. Maximum uncertainty before stopping the
            optimization in the predicted landscape.

        Returns
        -------
        Optimized atom structure.

        """

        self.fmax = fmax

        # Initialization. Evaluate one point if not evaluated
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

        converged_dimer(self)
        self.list_fmax = get_fmax(-np.array([self.list_gradients[-1]]),
                                  self.num_atoms)
        self.max_abs_forces = np.max(np.abs(self.list_fmax))
        self.list_max_abs_forces.append(self.max_abs_forces)
        print_info(self)

        success_hyper = False

        while not converged_dimer(self):

            # 1. Train Machine Learning model.
            train = np.copy(self.list_train)
            targets = np.copy(self.list_targets)
            gradients = np.copy(self.list_gradients)
            u_prior = np.max(targets[:, 0])
            scaled_targets = targets - u_prior
            sigma_f = 1e-3 + np.std(scaled_targets)**2

            if kernel == 'SQE':
                opt_hyper = True
                kdict = [{'type': 'gaussian', 'width': 0.4,
                          'dimension': 'single',
                          'bounds': ((0.1, 3.),),
                          'scaling': sigma_f,
                          'scaling_bounds': ((sigma_f, sigma_f),)},
                         {'type': 'noise_multi',
                          'hyperparameters': [0.005, 0.005 * 0.4**2],
                          'bounds': ((0.001, 0.050),
                                     (0.001 * 0.4**2, 0.050),)}
                         ]

            if kernel == 'SQE_scale':
                opt_hyper = True
                kdict = [{'type': 'gaussian', 'width': 0.4,
                          'dimension': 'single',
                          'bounds': ((0.4, 0.6),),
                          'scaling': sigma_f,
                          'scaling_bounds': ((sigma_f, sigma_f + 1e2),)},
                         {'type': 'noise_multi',
                          'hyperparameters': [0.005, 0.005 * 0.4**2],
                          'bounds': ((0.001, 0.050),
                                     (0.001 * 0.4**2, 0.050),)}
                         ]

                if success_hyper is not False:
                    kdict = success_hyper

            if self.index_mask is not None:
                train = apply_mask(list_to_mask=train,
                                   mask_index=self.index_mask)[1]
                gradients = apply_mask(list_to_mask=gradients,
                                       mask_index=self.index_mask)[1]

            print('Training a GP process...')
            print('Number of training points:', len(scaled_targets))

            self.gp = GaussianProcess(kernel_list=kdict,
                                      regularization=0.0,
                                      regularization_bounds=(0.0, 0.0),
                                      train_fp=train,
                                      train_target=scaled_targets,
                                      gradients=gradients,
                                      optimize_hyperparameters=False)

            print('GP process trained.')

            if opt_hyper is True:
                if self.feval > 5:
                    self.gp.optimize_hyperparameters()
                    print('Hyperparam. optimization:', self.gp.theta_opt)
                    if self.gp.theta_opt.success is True:
                        print('Hyperparam. optimization was successful.')
                        print('Updating kernel list...')
                        success_hyper = self.gp.kernel_list

                    if self.gp.theta_opt.success is not True:
                        print('Hyperparam. optimization unsuccessful.')
                        if success_hyper is False:
                            print('Not enough data...')
                        if success_hyper is not False:
                            print('Using the last optimized hyperparamters.')

                    print('GP process optimized.')
                    print('Kernel list:', self.gp.kernel_list)

            # 2. Optimize Machine Learning model.

            guess = self.ase_ini
            guess_pos = self.list_train[0]
            guess.positions = guess_pos.reshape(-1, 3)
            guess.set_calculator(ASECalc(gp=self.gp,
                                         index_constraints=self.index_mask,
                                         scaling_targets=u_prior)
                                 )
            guess.info['iteration'] = self.iter

            # Optimization using the dimer method:

            traj = Trajectory('gp_dimer_opt.traj', 'w', guess)
            traj.write()

            d_control = DimerControl(initial_eigenmode_method='displacement',
                                     displacement_method='vector',
                                     logfile=None,
                                     mask=dmask)
            d_atoms = MinModeAtoms(guess, d_control)
            d_atoms.displace(displacement_vector=vector)
            dim_rlx = MinModeTranslate(d_atoms,
                                       trajectory=traj,
                                       logfile='-')
            ml_converged = False

            while ml_converged is False:
                dim_rlx.run(fmax=fmax*0.9, steps=1)

                pos_ml = np.array(guess.positions).flatten()
                pos_ml = apply_mask([pos_ml], mask_index=self.index_mask)[1]
                pred_ml = self.gp.predict(test_fp=pos_ml, uncertainty=True)
                unc_ml = pred_ml['uncertainty_with_reg'][0]
                if unc_ml >= max_uncertainty:
                    print('Maximum uncertainty reach. Early stop.')
                    ml_converged = True
                if dim_rlx.converged():
                    print('ML optimized.')
                    ml_converged = True

            try:
                get_plot('gp_dimer_opt.traj', self.gp, scaling=u_prior)
            except:
                pass

            interesting_point = guess.get_positions().flatten()

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
            self.list_max_abs_forces.append(self.max_abs_forces)

            self.iter += 1
            print_info(self)

            # Maximum number of iterations reached.
            if self.iter >= steps:
                print('Not converged. Maximum number of iterations reached.')
                break


class ASECalc(Calculator):

    """
    CatLearn/ASE calculator.
    """

    implemented_properties = ['energy', 'forces']
    nolabel = True

    def __init__(self, gp, index_constraints, scaling_targets,
                 finite_step=1e-4, **kwargs):

        Calculator.__init__(self, **kwargs)

        self.gp = gp
        self.scaling = scaling_targets
        self.fs = finite_step
        self.ind_constraints = index_constraints

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):

        # Atoms object.
        self.atoms = atoms

        def pred_energy_test(test, gp=self.gp, scaling=self.scaling):

            # Get predictions.
            predictions = gp.predict(test_fp=test)
            energy = predictions['prediction'][0][0] + scaling
            return energy

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


def get_plot(filename, gp, scaling):

    """ Function for plotting each step of the toy model Muller-Brown .
    """

    fig = plt.figure(figsize=(10, 16))
    ax1 = plt.subplot()

    # Grid test points (x,y).
    x_lim = [-6., 6.]
    y_lim = [-6., 6.]

    # Axes.
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)

    # Range of energies:
    min_color = -0.1
    max_color = +15.

    # Length of the grid test points (nxn).
    test_points = 50

    test = []
    testx = np.linspace(x_lim[0], x_lim[1], test_points)
    testy = np.linspace(y_lim[0], y_lim[1], test_points)
    for i in range(len(testx)):
        for j in range(len(testy)):
            test1 = [testx[i], testy[j], 0.0]
            test.append(test1)
    test = np.array(test)

    x = []
    for i in range(len(test)):
        t = test[i][0]
        x.append(t)
    y = []
    for i in range(len(test)):
        t = test[i][1]
        y.append(t)

    # Plot real function.

    energy_ase = []
    for i in test:
        positions = np.array([[i[0], i[1], i[2]]])
        energy_structure = gp.predict(test_fp=positions)
        energy_ase.append(energy_structure['prediction'][0][0] + scaling)

    crange = np.linspace(min_color, max_color, 1000)

    zi = plt.mlab.griddata(x, y, energy_ase, testx, testy, interp='linear')

    image = ax1.contourf(testx, testy, zi, crange, alpha=1., cmap='Spectral_r',
                         extend='neither', antialiased=False)
    for c in image.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000001)

    crange2 = np.linspace(min_color, max_color, 10)
    ax1.contour(testx, testy, zi, crange2, alpha=0.6, antialiased=True)

    interval_colorbar = np.linspace(min_color, max_color, 5)

    fig.colorbar(image, ax=ax1, ticks=interval_colorbar, extend='None',
                 panchor=(0.5, 0.0),
                 orientation='horizontal', label='Function value (a.u.)')

    # Plot each point evaluated.
    all_structures = read(filename, ':')

    geometry_data = []
    for j in all_structures:
        geometry_data.append(j.get_positions().flatten())
    geometry_data = np.reshape(geometry_data, ((len(geometry_data)), 3))

    trained_points = gp.train_fp

    ax1.plot(geometry_data[:, 0], geometry_data[:, 1], c='black', lw=2.)
    ax1.scatter(trained_points[:, 0], trained_points[:, 1], marker='o',
                s=50.0, c='white', linewidths=1.0, edgecolors='black',
                alpha=1.0)

    plt.tight_layout(h_pad=1)
    plt.show()
