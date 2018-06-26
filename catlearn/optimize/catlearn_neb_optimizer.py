import numpy as np
from catlearn.optimize.warnings import *
from catlearn.optimize.io import ase_traj_to_catlearn, store_results_neb
from catlearn.optimize.ml_calculator import GPCalculator, train_ml_process
from catlearn.optimize.convergence import get_fmax
from catlearn.optimize.get_real_values import eval_and_append
from catlearn.optimize.catlearn_ase_calc import CatLearnASE
from catlearn.optimize.constraints import create_mask_ase_constraints
from catlearn.optimize.plots import get_plot_mullerbrown, get_plots_neb, \
                                    get_plot_mullerbrown_p
from ase.io.trajectory import TrajectoryWriter
from ase.neb import NEB
from ase.neb import NEBTools
from ase.io import read, write
from ase.optimize import MDMin, FIRE
from scipy.spatial import distance
import copy
import os
from scipy.interpolate import CubicSpline
from catlearn.optimize.io import array_to_ase


class CatLearnNEB(object):

    def __init__(self, start, end, path=None, n_images=None,
                 spin_polarized=False,
                 interpolation=None, mic=False,
                 neb_method='improvedtangent', spring=None,
                 ml_calc=None, ase_calc=None, inc_prev_calcs=False,
                 stabilize=False, restart=False):
        """ Nudged elastic band (NEB) setup.

        Parameters
        ----------
        start: Trajectory file (in ASE format).
            Initial end-point of the NEB path.
        end: Trajectory file (in ASE format).
            Final end-point of the NEB path.
        path: Trajectory file (in ASE format) (optional).
            Atoms trajectory with the intermediate images of the path
            between the two end-points.
        n_images: int
            Number of images of the path (if not included a path before).
             The number of images include the 2 end-points of the NEB path.
        spring: float
            Spring constant(s) in eV/Ang.
        interpolation: string
            Automatic interpolation can be done ('idpp' and 'linear' as
            implemented in ASE).
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html.
        neb_method: string
            NEB method as implemented in ASE. ('aseneb', 'improvedtangent'
            or 'eb').
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html.
        ml_calc : ML calculator Object.
            Machine Learning calculator (e.g. Gaussian Process). Default is GP.
        ase_calc: ASE calculator Object.
            ASE calculator as implemented in ASE.
            See https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
        """

        # Start end-point, final end-point and path (optional):
        self.start = start
        self.end = end
        self.n_images = n_images

        # General setup:
        self.iter = 0
        self.feval = 0
        self.ml_calc = ml_calc
        self.ase_calc = ase_calc
        self.ase = True
        self.spin = spin_polarized
        self.mic = mic

        # Reset:
        self.constraints = None
        self.interesting_point = None

        # Create new file to store warnings and errors:
        open('warnings_and_errors.txt', 'w')

        assert start is not None, err_not_neb_start()
        assert end is not None, err_not_neb_end()
        assert self.ase_calc, err_not_ase_calc_traj()

        # A) Include previous calculations for training the ML model.
        is_endpoint = read(start, ':')
        fs_endpoint = read(end, ':')

        # B) Only include initial and final (optimized) images.
        if inc_prev_calcs is False:
            is_endpoint = read(start, '-1:')
            fs_endpoint = read(end, '-1:')
        is_pos = is_endpoint[-1].get_positions().flatten()
        fs_pos = fs_endpoint[-1].get_positions().flatten()

        # A and B) Restart mode.

        # Read the previously evaluated structures and append them to the
        # training list
        if restart is True:
            restart_filename = 'evaluated_structures.traj'
            if not os.path.isfile(restart_filename):
                warning_restart_neb()
            if os.path.isfile(restart_filename):
                evaluated_images = read(restart_filename, ':')
                is_endpoint = evaluated_images + is_endpoint
                stabilize = False

        # Write previous evaluated images in evaluations:
        write('./evaluated_structures.traj',
              is_endpoint + fs_endpoint)

        # Obtain the energy of the endpoints for scaling:
        energy_is = is_endpoint[-1].get_potential_energy()
        energy_fs = fs_endpoint[-1].get_potential_energy()

        # Set scaling of the targets:
        self.scale_targets = np.min([energy_is, energy_fs])

        # Convert atoms information into data to feed the ML process.
        if os.path.exists('./tmp.traj'):
                os.remove('./tmp.traj')
        merged_trajectory = is_endpoint + fs_endpoint
        write('tmp.traj', merged_trajectory)

        trj = ase_traj_to_catlearn(traj_file='tmp.traj')
        self.list_train, self.list_targets, self.list_gradients, trj_images,\
            self.constraints, self.num_atoms = [trj['list_train'],
                                                trj['list_targets'],
                                                trj['list_gradients'],
                                                trj['images'],
                                                trj['constraints'],
                                                trj['num_atoms']]
        os.remove('./tmp.traj')
        self.ase_ini = trj_images[0]
        self.num_atoms = len(self.ase_ini)
        if len(self.constraints) < 0:
            self.constraints = None
        if self.constraints is not None:
            self.ind_mask_constr = create_mask_ase_constraints(
                                                self.ase_ini, self.constraints)

        # Configure ML calculator.
        if self.ml_calc is None:
            self.kdict = {'k1': {'type': 'gaussian', 'width': 0.5,
                                 'dimension': 'single',
                                 'bounds': ((0.05, 1.0), ),
                                 'scaling': 1.0,
                                 'scaling_bounds': ((1.0, 1.0), )}
                          }

            self.ml_calc = GPCalculator(
                kernel_dict=self.kdict, opt_hyperparam=True, scale_data=False,
                scale_optimizer=False, calc_uncertainty=True,
                regularization=1e-5, regularization_bounds=(1e-5, 1e-5))

        # Settings for the NEB.
        self.neb_method = neb_method
        self.spring = spring
        self.initial_endpoint = is_endpoint[-1]
        self.final_endpoint = fs_endpoint[-1]

        # A) Create images using interpolation if user do not feed a path:
        self.d_start_end = np.abs(distance.euclidean(is_pos, fs_pos))

        if path is None:
            if self.spring is None:
                self.spring = np.sqrt((self.n_images-1) / self.d_start_end)
            self.images = create_ml_neb(is_endpoint=self.initial_endpoint,
                                        fs_endpoint=self.final_endpoint,
                                        images_interpolation=None,
                                        n_images=self.n_images,
                                        constraints=self.constraints,
                                        index_constraints=self.ind_mask_constr,
                                        trained_process=None,
                                        ml_calculator=self.ml_calc,
                                        scaling_targets=self.scale_targets,
                                        iteration=self.iter
                                        )

            neb_interpolation = NEB(self.images, k=self.spring)

            neb_interpolation.interpolate(method=interpolation, mic=self.mic)

            self.initial_images = copy.deepcopy(self.images)

        # B) If the user sets a path:
        if path is not None:
            images_path = read(path, ':')

            if not np.array_equal(images_path[0].get_positions().flatten(),
                                  is_pos):
                images_path.insert(0, self.initial_endpoint)
            if not np.array_equal(images_path[-1].get_positions().flatten(),
                                  fs_pos):
                images_path.append(self.final_endpoint)

            self.n_images = len(images_path)
            self.images = create_ml_neb(is_endpoint=self.initial_endpoint,
                                        fs_endpoint=self.final_endpoint,
                                        images_interpolation=images_path,
                                        n_images=self.n_images,
                                        constraints=self.constraints,
                                        index_constraints=self.ind_mask_constr,
                                        trained_process=None,
                                        ml_calculator=self.ml_calc,
                                        scaling_targets=self.scale_targets,
                                        iteration=self.iter
                                        )

        # Save files with all the paths tested:
        write('all_predicted_paths.traj', self.images)

        if stabilize is True:
            for i in self.images[1:-1]:
                interesting_point = i.get_positions().flatten()
                eval_and_append(self, interesting_point)
                TrajectoryWriter(atoms=self.ase_ini,
                                 filename='./evaluated_structures.traj',
                                 mode='a').write()
                self.iter = 0

        self.uncertainty_path = np.zeros(len(self.images))

        # Stabilize spring constant:
        if self.spring is None:
            self.spring = np.sqrt((self.n_images-1) / self.d_start_end)

        # Get path distance:
        self.path_distance = copy.deepcopy(self.d_start_end)

        # Check the magnetic moments of the initial and final states:

        if self.spin is True:
            warning_spin_neb()
            is_spin = copy.deepcopy(is_endpoint[-1])
            is_spin.set_calculator(self.ase_calc)
            is_spin.get_potential_energy()
            self.magmom_is = []
            for i in is_spin:
                self.magmom_is.append(i.magmom)
            fs_spin = copy.deepcopy(fs_endpoint[-1])
            fs_spin.set_calculator(self.ase_calc)
            fs_spin.get_potential_energy()
            self.magmom_fs = []
            for i in fs_spin:
                self.magmom_fs.append(i.magmom)
            self.magmom_is = np.array(self.magmom_is)
            self.magmom_fs = np.array(self.magmom_fs)
            print('\nInitial magmoms:')
            print('---------------------------------------------------------')
            print('Initial state end-point:',
                                 is_endpoint[-1].get_initial_magnetic_moments()
)
            print('Final state end-point:',
                                fs_endpoint[-1].get_initial_magnetic_moments())
            print('\nOptimized magmoms:')
            print('---------------------------------------------------------')
            print('Initial state end-point:', self.magmom_is)
            print('Final state end-point:', self.magmom_fs)
            print('\n')


    def run(self, fmax=0.05, unc_convergence=0.010, max_iter=500,
            ml_algo='MDMin', ml_max_iter=500, plot_neb_paths=False):

        """Executing run will start the optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angs).
        unc_convergence: float
            Maximum uncertainty for convergence (in eV).
        max_iter : int
            Maximum number of iterations in the surrogate model.
        ml_algo : string
            Algorithm for the surrogate model. Implemented are:
            'MDMin' and 'FIRE' as implemented in ASE.
            See https://wiki.fysik.dtu.dk/ase/ase/optimize.html
        ml_max_iter : int
            Maximum number of ML NEB iterations.
        plot_neb_paths: bool
            If True it prints and stores (in csv format) the last predicted
            NEB path obtained by the surrogate ML model. Note: Python package
            matplotlib is required.

        Returns
        -------
        NEB optimized path.
        Files :
        """

        while True:

            # 1) Train Machine Learning process:

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
                                       scaling_targets=self.scale_targets)

            trained_process = process['trained_process']
            ml_calc = process['ml_calc']
            print('ML process trained.')

            # 2) Setup and run ML NEB:

            if self.iter > 0:

                # Redistribute images along the path.

                self.images = redistribute_images_path(
                                              images=self.images,
                                              d_images=s,
                                              path_distance=self.path_distance,
                                              n_images=self.n_images,
                                              n_atoms=self.num_atoms
                                              )

                # If the previous run didn't converge use the initial path.
                if neb_opt.__dict__['nsteps'] >= ml_max_iter - 2:
                    self.images = copy.deepcopy(self.initial_images)

            starting_path = self.images

            self.images = create_ml_neb(is_endpoint=self.initial_endpoint,
                                        fs_endpoint=self.final_endpoint,
                                        images_interpolation=starting_path,
                                        n_images=self.n_images,
                                        constraints=self.constraints,
                                        index_constraints=self.ind_mask_constr,
                                        trained_process=trained_process,
                                        ml_calculator=ml_calc,
                                        scaling_targets=self.scale_targets,
                                        iteration=self.iter
                                        )

            ml_neb = NEB(self.images, climb=False,
                         method=self.neb_method,
                         k=self.spring)

            neb_opt = eval(ml_algo)(ml_neb, dt=0.1)

            print('Starting ML NEB optimization...')
            neb_opt.run(fmax=fmax,
                        steps=ml_max_iter)

            # if np.max(self.uncertainty_path[1:-1]) <= 2 * unc_convergence:
            print('Starting ML NEB optimization using climbing image...')
            ml_neb = NEB(self.images, climb=True,
                         method=self.neb_method,
                         k=self.spring)

            neb_opt = eval(ml_algo)(ml_neb, dt=0.1)
            neb_opt.run(fmax=fmax, steps=ml_max_iter)
            print('ML NEB optimized.')

            # 3) Get results from ML NEB using ASE NEB tools:
            # See https://wiki.fysik.dtu.dk/ase/ase/neb.html

            interesting_point = []

            # Get fit of the discrete path.
            neb_tools = NEBTools(self.images)
            [s, e, sfit, efit] = neb_tools.get_fit()[0:4]

            self.path_distance = s[-1]

            self.uncertainty_path = []
            energies_path = []
            for i in self.images:
                self.uncertainty_path.append(i.info['uncertainty'])
                energies_path.append(i.get_total_energy())

            # Extra: Spin polarized calculations. Interpolation between is/fs.
            if self.spin is True:
                vect_magmom = (self.magmom_fs-self.magmom_is) / s[-1]
                for i in range(0, len(self.images)):
                    self.images[i].__dict__['arrays']['initial_magmoms'] =\
                                            (vect_magmom*s[i] + self.magmom_is)

            # Select image with maximum uncertainty.
            if self.iter % 2 == 0:
                argmax_unc = np.argmax(self.uncertainty_path[1:-1])

                interesting_image = self.images[1:-1][int(argmax_unc)]
                interesting_point = interesting_image.get_positions().flatten()
                interesting_magmom = \
                    interesting_image.get_initial_magnetic_moments()

            # Select image with max. predicted value (absolute value).
            if self.iter % 2 == 1:
                argmax_unc = np.argmax(np.abs(energies_path[1:-1]))
                interesting_image = self.images[1:-1][int(argmax_unc)]
                interesting_point = interesting_image.get_positions().flatten()
                interesting_magmom = \
                    interesting_image.get_initial_magnetic_moments()

            # Plots results in each iteration.
            if plot_neb_paths is True:
                get_plots_neb(images=self.images,
                              selected=int(argmax_unc), iter=self.iter)

            if plot_neb_paths is True:
                if self.ase_calc.__dict__['name'] == 'mullerbrown':
                    get_plot_mullerbrown(images=self.images,
                                         interesting_point=interesting_point,
                                         trained_process=trained_process,
                                         list_train=self.list_train)
                    # get_plot_mullerbrown_p(images=self.images,
                    #                        interesting_point=interesting_point,
                    #                        trained_process=trained_process,
                    #                        list_train=self.list_train)
            # Store results each iteration:
            store_results_neb(s, e, sfit, efit, self.uncertainty_path)

            # 3) Add a new training point and evaluate it.
            eval_and_append(self, interesting_point, interesting_magmom)

            # 4) Store results.

            # Evaluated images.
            TrajectoryWriter(atoms=self.ase_ini,
                             filename='./evaluated_structures.traj',
                             mode='a').write()
            # Last path.
            write('last_predicted_path.traj', self.images)

            # All paths.
            for i in self.images:
                TrajectoryWriter(atoms=i,
                                 filename='all_predicted_paths.traj',
                                 mode='a').write()

            print('Length of initial path (Angstrom):', self.d_start_end)
            print('Length of the current path (Angstrom):', self.path_distance)
            print('Spring constant (eV/Angstrom):', self.spring)
            print('Max. uncertainty (eV):',
                  np.max(self.uncertainty_path[1:-1]))
            print('Image #id with max. uncertainty:',
                  np.argmax(self.uncertainty_path[1:-1]) + 2)
            print('Number of iterations:', self.iter)

            # Break if converged:

            max_forces = get_fmax(-np.array([self.list_gradients[-1]]),
                                  self.num_atoms)
            max_abs_forces = np.max(np.abs(max_forces))

            print('Max. force of the last image evaluated (eV/Angstrom):',
                  max_abs_forces)
            print('Energy of the last image evaluated (eV):',
                  self.list_targets[-1][0])
            print('Energy of the last image evaluated w.r.t. to endpoint ('
                  'eV):', self.list_targets[-1][0] - self.scale_targets)
            print('Number #id of the last evaluated image:', argmax_unc + 2)

            # Update spring:
            self.spring = np.sqrt((self.n_images-1) / s[-1])

            # 5) Check convergence:

            # Check whether the evaluated point is a stationary point.
            if max_abs_forces <= fmax:
                congrats_stationary_neb()
                if np.max(self.uncertainty_path[1:-1]) < unc_convergence:
                    congrats_neb_converged()
                    break

            # Break if reaches the max number of iterations set by the user.
            if max_iter <= self.iter:
                warning_max_iter_reached()
                break

        # Print Final convergence:
        print('Number of function evaluations in this run:', self.iter)


def create_ml_neb(is_endpoint, fs_endpoint, images_interpolation,
                  n_images, constraints, index_constraints, trained_process,
                  ml_calculator, scaling_targets, iteration):

    # End-points of the NEB path:
    s_guess_ml = copy.deepcopy(is_endpoint)
    f_guess_ml = copy.deepcopy(fs_endpoint)

    # Create ML NEB path:
    imgs = [s_guess_ml]

    # Scale energies (initial):
    imgs[0].__dict__['_calc'].__dict__['results']['energy'] = \
        imgs[0].__dict__['_calc'].__dict__['results']['energy'] - \
        scaling_targets

    # Append labels, uncertainty and iter to the first end-point:
    imgs[0].info['label'] = 0
    imgs[0].info['uncertainty'] = 0.0
    imgs[0].info['iteration'] = iteration

    for i in range(1, n_images-1):
        image = copy.deepcopy(s_guess_ml)
        image.info['label'] = i
        image.info['uncertainty'] = 0.0
        image.info['iteration'] = iteration
        image.set_calculator(CatLearnASE(trained_process=trained_process,
                                         ml_calc=ml_calculator,
                                         index_constraints=index_constraints
                                         ))
        if images_interpolation is not None:
            image.set_positions(images_interpolation[i].get_positions())
        image.set_constraint(constraints)
        imgs.append(image)

    # Scale energies (final):
    imgs.append(f_guess_ml)
    imgs[-1].__dict__['_calc'].__dict__['results']['energy'] = \
        imgs[-1].__dict__['_calc'].__dict__['results']['energy'] - \
        scaling_targets

    # Append labels, uncertainty and iter to the last end-point:
    imgs[-1].info['label'] = n_images - 1
    imgs[-1].info['uncertainty'] = 0.0
    imgs[-1].info['iteration'] = iteration

    return imgs


def redistribute_images_path(images, d_images, path_distance, n_images,
                             n_atoms):
    x = d_images
    y = []

    for i in images:
        y.append(i.get_positions().flatten())
    cubic = CubicSpline(x, y)
    lin_space = np.linspace(0, path_distance, n_images)
    reag_pos = cubic(lin_space)

    for i in range(0, len(images)):
        pos = array_to_ase(reag_pos[i], n_atoms)
        images[i].set_positions(pos)
    return images
