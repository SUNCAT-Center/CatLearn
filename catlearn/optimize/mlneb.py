import numpy as np
from catlearn.optimize.warnings import *
from catlearn.optimize.io import ase_traj_to_catlearn, store_results_neb, \
                                 print_version, store_trajectory_neb, \
                                 print_info_neb
from catlearn.optimize.convergence import get_fmax
from catlearn.optimize.get_real_values import eval_and_append
from catlearn.optimize.constraints import create_mask, apply_mask
from ase.neb import NEB
from ase.neb import NEBTools
from ase.io import read, write
from ase.optimize import MDMin
from scipy.spatial import distance
import copy
import os
from catlearn.regression import GaussianProcess
from ase.calculators.calculator import Calculator, all_changes
from ase.atoms import Atoms
from catlearn import __version__

class MLNEB(object):

    def __init__(self, start, end, prev_calculations=None,
                 n_images=0.25, k=None, interpolation='idpp', mic=False,
                 neb_method='improvedtangent', ase_calc=None, restart=True):

        """ Nudged elastic band (NEB) setup.

        Parameters
        ----------
        start: Trajectory file (in ASE format) or Atoms object.
            Initial end-point of the NEB path or Atoms object.
        end: Trajectory file (in ASE format).
            Final end-point of the NEB path.
        n_images: int or float
            Number of images of the path (if not included a path before).
             The number of images include the 2 end-points of the NEB path.
        k: float or list
            Spring constant(s) in eV/Ang.
        interpolation: string or Atoms list or Trajectory
            Automatic interpolation can be done ('idpp' and 'linear' as
            implemented in ASE).
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html.
            Manual: Trajectory file (in ASE format) or list of Atoms.
            Atoms trajectory or list of Atoms containing the images along the
            path.
        neb_method: string
            NEB method as implemented in ASE. ('aseneb', 'improvedtangent'
            or 'eb').
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html.
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

        path = None

        # Convert Atoms and list of Atoms to trajectory files.
        if isinstance(start, Atoms):
            write('initial.traj', start)
            start = 'initial.traj'
        if isinstance(end, Atoms):
            write('final.traj', end)
            end = 'final.traj'
        if interpolation != 'idpp' and interpolation != 'linear':
            path = interpolation
        if isinstance(path, list):
            write('initial_path.traj', path)
            path = 'initial_path.traj'
        if isinstance(prev_calculations, list):
            write('prev_calcs.traj', prev_calculations)
            prev_calculations = 'prev_calcs.traj'

        # Prevent duplicates:
        if prev_calculations is not None:
            restart = False

        # Start end-point, final end-point and path (optional).
        self.start = start
        self.end = end
        self.n_images = n_images
        self.feval = 0

        # General setup.
        self.iter = 0
        self.ase_calc = ase_calc
        self.ase = True
        self.mic = mic
        self.version = 'ML-NEB ' + __version__
        print_version(self.version)

        # Reset.
        self.constraints = None
        self.interesting_point = None
        self.acq = None
        self.gp = None

        # Create new file to store warnings and errors.
        open('warnings_and_errors.txt', 'w')

        assert start is not None, err_not_neb_start()
        assert end is not None, err_not_neb_end()
        assert self.ase_calc, err_not_ase_calc_traj()

        # A) Include previous calculations for training the ML model.
        if prev_calculations is not None:
            prev_calculations = read(prev_calculations, ':')
            is_endpoint_prev_calcs = read(start, '-1:')
            is_endpoint = prev_calculations + is_endpoint_prev_calcs
            fs_endpoint = read(end, '-1:')

        # B) Only include initial and final (optimized) images.
        if prev_calculations is None:
            is_endpoint = read(start, '-1:')
            fs_endpoint = read(end, '-1:')
        is_pos = is_endpoint[-1].get_positions().flatten()
        fs_pos = fs_endpoint[-1].get_positions().flatten()

        # Check the magnetic moments of the initial and final states:
        self.magmom_is = is_endpoint[-1].get_initial_magnetic_moments()
        self.magmom_fs = fs_endpoint[-1].get_initial_magnetic_moments()

        if not np.array_equal(self.magmom_is, np.zeros_like(self.magmom_is)):
            warning_spin_neb()
        if not np.array_equal(self.magmom_fs, np.zeros_like(self.magmom_fs)):
            warning_spin_neb()

        # Convert atoms information into data to feed the ML process.

        # Include Restart mode.

        if restart is not True:
            if os.path.exists('./tmp.traj'):
                    os.remove('./tmp.traj')
            merged_trajectory = is_endpoint + fs_endpoint
            write('tmp.traj', merged_trajectory)
            trj = ase_traj_to_catlearn(traj_file='tmp.traj')
            os.remove('./tmp.traj')
            write('./evaluated_structures.traj', is_endpoint + fs_endpoint)

        if restart is True:
            eval_file = 'evaluated_structures.traj'
            if os.path.exists(eval_file):
                trj = ase_traj_to_catlearn(traj_file=eval_file)
            if not os.path.exists(eval_file):
                if os.path.exists('./tmp.traj'):
                        os.remove('./tmp.traj')
                merged_trajectory = is_endpoint + fs_endpoint
                write('tmp.traj', merged_trajectory)
                trj = ase_traj_to_catlearn(traj_file='tmp.traj')
                os.remove('./tmp.traj')
                write('./evaluated_structures.traj', is_endpoint + fs_endpoint)

        self.list_train, self.list_targets, self.list_gradients, trj_images,\
            self.constraints, self.num_atoms = [trj['list_train'],
                                                trj['list_targets'],
                                                trj['list_gradients'],
                                                trj['images'],
                                                trj['constraints'],
                                                trj['num_atoms']]
        self.ase_ini = trj_images[0]
        self.num_atoms = len(self.ase_ini)
        if len(self.constraints) < 0:
            self.constraints = None
        if self.constraints is not None:
            self.index_mask = create_mask(self.ase_ini, self.constraints)

        # Obtain the energy of the endpoints for scaling:
        self.energy_is = is_endpoint[-1].get_potential_energy()
        self.energy_fs = fs_endpoint[-1].get_potential_energy()

        # Set scaling of the targets:
        self.max_targets = np.max([self.energy_is, self.energy_fs])

        # Settings for the NEB.
        self.neb_method = neb_method
        self.spring = k
        self.initial_endpoint = is_endpoint[-1]
        self.final_endpoint = fs_endpoint[-1]

        # A) Create images using interpolation if user do not feed a path:
        if path is None:
            self.d_start_end = np.abs(distance.euclidean(is_pos, fs_pos))
            if isinstance(self.n_images, float):
                self.n_images = int(self.d_start_end/self.n_images)
                if self. n_images <= 3:
                    self.n_images = 3
            if self.spring is None:
                self.spring = np.sqrt((self.n_images-1) / self.d_start_end)
            self.images = create_ml_neb(is_endpoint=self.initial_endpoint,
                                        fs_endpoint=self.final_endpoint,
                                        images_interpolation=None,
                                        n_images=self.n_images,
                                        constraints=self.constraints,
                                        index_constraints=self.index_mask,
                                        scaling_targets=self.max_targets,
                                        iteration=self.iter,
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
                                        index_constraints=self.index_mask,
                                        scaling_targets=self.max_targets,
                                        iteration=self.iter,
                                        )
            self.d_start_end = np.abs(distance.euclidean(is_pos, fs_pos))

        # Save files with all the paths that have been predicted:
        write('all_predicted_paths.traj', self.images)

        self.uncertainty_path = np.zeros(len(self.images))

        # Guess spring constant if spring was not set by the user:
        if self.spring is None:
            self.spring = np.sqrt(self.n_images-1) / self.d_start_end

        # Get initial path distance:
        self.path_distance = copy.deepcopy(self.d_start_end)

        # Get forces for the previous steps
        self.list_max_abs_forces = []
        for i in self.list_gradients:
                self.list_fmax = get_fmax(-np.array([i]), self.num_atoms)
                self.max_abs_forces = np.max(np.abs(self.list_fmax))
                self.list_max_abs_forces.append(self.max_abs_forces)

        print_info_neb(self)

    def run(self, fmax=0.05, unc_convergence=0.050, steps=200,
            trajectory='ML_NEB_catlearn.traj', acquisition='acq_2', dt=0.025):

        """Executing run will start the optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angs).
        unc_convergence: float
            Maximum uncertainty for convergence (in eV).
        steps : int
            Maximum number of iterations in the surrogate model.
        trajectory: string
            Filename to store the output.
        acquisition : string
            Acquisition function.
        dt : float
            dt parameter for MDMin.

        Returns
        -------
        NEB optimized path.
        Files :
        """
        self.acq = acquisition

        # Calculate a third point if only known initial & final structures.
        if len(self.list_targets) == 2:
            middle = int(self.n_images * (2./3.))
            if self.energy_is >= self.energy_fs:
                middle = int(self.n_images * (1./3.))
            self.interesting_point = \
                self.images[middle].get_positions().flatten()
            eval_and_append(self, self.interesting_point)
            self.iter += 1
            self.max_forces = get_fmax(-np.array([self.list_gradients[-1]]),
                                   self.num_atoms)
            self.max_abs_forces = np.max(np.abs(self.max_forces))
            self.list_max_abs_forces.append(self.max_abs_forces)
            print_info_neb(self)
        stationary_point_found = False

        while True:

            # 1. Train Machine Learning process.
            train_gp_model(self)

            # 2. Setup and run ML NEB:

            ml_steps = self.n_images * len(self.index_mask)
            ml_steps = 250 if ml_steps <= 250 else ml_steps  # Min steps.
            ml_steps = 750 if ml_steps >= 750 else ml_steps  # Min steps.

            print('Max number steps:', ml_steps)
            ml_cycles = 0

            while True:

                starting_path = self.images  # Start from last path.

                if ml_cycles == 0:
                    sp = '0:' + str(self.n_images)
                    print('Using initial path.')
                    starting_path = read('./all_predicted_paths.traj', sp)

                if ml_cycles == 1:
                    print('Using last predicted path.')
                    sp = str(-self.n_images*1) + ':' + str(-1)
                    starting_path = read('./all_predicted_paths.traj', sp)

                self.images = create_ml_neb(is_endpoint=self.initial_endpoint,
                                            fs_endpoint=self.final_endpoint,
                                            images_interpolation=starting_path,
                                            n_images=self.n_images,
                                            constraints=self.constraints,
                                            index_constraints=self.index_mask,
                                            gp=self.gp,
                                            scaling_targets=self.max_target,
                                            iteration=self.iter)

                ml_neb = NEB(self.images, climb=True,
                             method=self.neb_method,
                             k=self.spring)

                print('Optimizing ML CI-NEB using dt:', dt)
                neb_opt = MDMin(ml_neb, dt=dt)
                neb_opt.run(fmax=(fmax * 0.9), steps=ml_steps)
                n_steps_performed = neb_opt.__dict__['nsteps']

                if n_steps_performed <= ml_steps-1:
                    print('Converged optimization in the predicted landscape.')
                    break

                ml_cycles += 1
                print('ML cycles performed:', ml_cycles)

                if ml_cycles == 2:
                    self.images = read('./last_predicted_path.traj', ':')
                    print('ML process not optimized...not safe...')
                    break

            # 3. Get results from ML NEB using ASE NEB Tools:
            # See https://wiki.fysik.dtu.dk/ase/ase/neb.html

            self.interesting_point = []

            # Get fit of the discrete path.
            get_results_predicted_path(self)

            pred_plus_unc = np.array(self.e_path[1:-1]) + np.array(
                                                   self.uncertainty_path[1:-1])

            # 4. Select next point to train (acquisition function):

            # Acquisition function 1:
            if self.acq == 'acq_1':
                # Behave like acquisition 4...
                # Select image with max. uncertainty.
                if self.iter % 2 == 0:
                    self.argmax_unc = np.argmax(self.uncertainty_path[1:-1])
                    self.interesting_point = self.images[1:-1][
                                    self.argmax_unc].get_positions().flatten()

                # Select image with max. predicted value.
                if self.iter % 2 == 1:
                    self.argmax_unc = np.argmax(pred_plus_unc)
                    self.interesting_point = self.images[1:-1][
                                int(self.argmax_unc)].get_positions().flatten()

            # Acquisition function 2:
            if self.acq == 'acq_2':
                # Select image with max. uncertainty.
                self.argmax_unc = np.argmax(self.uncertainty_path[1:-1])
                self.interesting_point = self.images[1:-1][
                                  self.argmax_unc].get_positions().flatten()

                # Select image with max. predicted value.
                if np.max(self.uncertainty_path[1:-1]) < unc_convergence:

                    self.argmax_unc = np.argmax(pred_plus_unc)
                    self.interesting_point = self.images[1:-1][
                                int(self.argmax_unc)].get_positions().flatten()
            # Acquisition function 3:
            if self.acq == 'acq_3':
                # Select image with max. uncertainty.
                self.argmax_unc = np.argmax(self.uncertainty_path[1:-1])
                self.interesting_point = self.images[1:-1][
                                    self.argmax_unc].get_positions().flatten()

                # When reached certain uncertainty apply acq. 1.
                if np.max(self.uncertainty_path[1:-1]) < unc_convergence:
                    # Select image with max. uncertainty.
                    if self.iter % 2 == 0:
                        self.argmax_unc = \
                                        np.argmax(self.uncertainty_path[1:-1])
                        self.interesting_point = self.images[1:-1][
                                    self.argmax_unc].get_positions().flatten()
                    # Select image with max. predicted value.
                    if self.iter % 2 == 1:
                        self.argmax_unc = np.argmax(pred_plus_unc)
                        self.interesting_point = self.images[1:-1][
                                int(self.argmax_unc)].get_positions().flatten()

            # Acquisition function 4 (from acq 2):
            if self.acq == 'acq_4':
                # Select image with max. uncertainty.
                if self.iter % 2 == 0:
                    self.argmax_unc = np.argmax(self.uncertainty_path[1:-1])
                    self.interesting_point = self.images[1:-1][
                                    self.argmax_unc].get_positions().flatten()

                # Select image with max. predicted value.
                if self.iter % 2 == 1:
                    self.argmax_unc = np.argmax(pred_plus_unc)
                    self.interesting_point = self.images[1:-1][
                                int(self.argmax_unc)].get_positions().flatten()
                # If stationary point is found behave like acquisition 2...
                if stationary_point_found is True:
                    # Select image with max. uncertainty.
                    self.argmax_unc = np.argmax(self.uncertainty_path[1:-1])
                    self.interesting_point = self.images[1:-1][
                    ml_cycles           .argmax_unc].get_positions().flatten()

                    # Select image with max. predicted value.
                    if np.max(self.uncertainty_path[1:-1]) < unc_convergence:

                        self.argmax_unc = np.argmax(pred_plus_unc)
                        self.interesting_point = self.images[1:-1][
                                int(self.argmax_unc)].get_positions().flatten()
            # Acquisition function 5 (From acq 3):
            if self.acq == 'acq_5':
                # Select image with max. uncertainty.
                self.argmax_unc = np.argmax(self.uncertainty_path[1:-1])
                self.interesting_point = self.images[1:-1][
                                  self.argmax_unc].get_positions().flatten()

                # When reached certain uncertainty apply acq. 1.
                if np.max(self.uncertainty_path[1:-1]) < unc_convergence:
                    # Select image with max. uncertainty.
                    if self.iter % 2 == 0:
                        self.argmax_unc = \
                                        np.argmax(self.uncertainty_path[1:-1])
                        self.interesting_point = self.images[1:-1][
                                    self.argmax_unc].get_positions().flatten()

                    # Select image with max. predicted value.
                    if self.iter % 2 == 1:
                        self.argmax_unc = np.argmax(pred_plus_unc)
                        self.interesting_point = self.images[1:-1][
                                int(self.argmax_unc)].get_positions().flatten()
                    # If stationary point is found behave like acquisition 2...
                    if stationary_point_found is True:
                        # Select image with max. uncertainty.
                        self.argmax_unc = \
                                         np.argmax(self.uncertainty_path[1:-1])
                        self.interesting_point = self.images[1:-1][
                                    self.argmax_unc].get_positions().flatten()

                    # Select image with max. predicted value.
                    if np.max(self.uncertainty_path[1:-1]) < unc_convergence:

                        self.argmax_unc = np.argmax(pred_plus_unc)
                        self.interesting_point = self.images[1:-1][
                                                  int(self.argmax_unc)].get_positions(
                                                  ).flatten()

            # 5. Add a new training point and evaluate it.
            print('Performing evaluation on the real landscape...')
            eval_and_append(self, self.interesting_point)
            self.iter += 1

            # 6. Store results.
            print('Energy of the last image evaluated (eV):',
                  self.list_targets[-1][0])

            self.energy_forward = np.max(self.e_path) - self.e_path[0]
            self.energy_backward = np.max(self.e_path) - self.e_path[-1]
            self.max_forces = get_fmax(-np.array([self.list_gradients[-1]]),
                                       self.num_atoms)
            self.max_abs_forces = np.max(np.abs(self.max_forces))

            print_info_neb(self)
            store_results_neb(self)
            store_trajectory_neb(self)

            # 7. Check convergence:

            if self.max_abs_forces <= fmax:
                stationary_point_found = True

            # Check whether the evaluated point is a stationary point.
            if self.max_abs_forces <= fmax:
                congrats_stationary_neb()

                if np.max(self.uncertainty_path[1:-1]) < unc_convergence:
                    # Save results of the final step (converged):
                    train_gp_model(self)
                    get_results_predicted_path(self)
                    store_results_neb(self)
                    congrats_neb_converged()
                    # Last path.
                    write(trajectory, self.images)
                    print('The optimized predicted path can be found in: ',
                          trajectory)
                    # Clean up:
                    os.remove('./last_predicted_path.traj')
                    os.remove('./all_predicted_paths.traj')
                    break

            # Break if reaches the max number of iterations set by the user.
            if steps <= self.iter:
                warning_max_iter_reached()
                break

        print('Number of steps performed in this run:', self.iter)


def create_ml_neb(is_endpoint, fs_endpoint, images_interpolation,
                  n_images, constraints, index_constraints,
                  scaling_targets, iteration, gp=None):

    # Create ML NEB path:
    imgs = [is_endpoint]

    # Append labels, uncertainty and iter to the first end-point:
    imgs[0].info['label'] = 0
    imgs[0].info['uncertainty'] = 0.0
    imgs[0].info['iteration'] = iteration

    for i in range(1, n_images-1):
        image = is_endpoint.copy()
        image.info['label'] = i
        image.info['uncertainty'] = 0.0
        image.info['iteration'] = iteration
        image.set_calculator(ASECalc(gp=gp,
                                     index_constraints=index_constraints,
                                     scaling_targets=scaling_targets))
        if images_interpolation is not None:
            image.set_positions(images_interpolation[i].get_positions())
        image.set_constraint(constraints)
        imgs.append(image)

    # Scale energies (final):
    imgs.append(fs_endpoint)

    # Append labels, uncertainty and iter to the last end-point:
    imgs[-1].info['label'] = n_images-1
    imgs[-1].info['uncertainty'] = 0.0
    imgs[-1].info['iteration'] = iteration

    return imgs


def train_gp_model(self):
    """
    Train GP Process
    """
    self.max_target = np.max(self.list_targets)
    scaled_targets = self.list_targets.copy() - self.max_target

    dimension = 'single'
    bounds = ((0.01, self.path_distance),)

    width = self.path_distance / 2

    noise_energy = 0.005
    noise_forces = 0.0005

    kdict = [{'type': 'gaussian', 'width': width,
              'dimension': dimension,
              'bounds': bounds,
              'scaling': 1.,
              'scaling_bounds': ((1., 1.),)},
             {'type': 'noise_multi',
              'hyperparameters': [noise_energy, noise_forces],
              'bounds': ((0.001, 0.010),
                         (0.0001, 0.0010),)}
             ]

    train = self.list_train.copy()
    gradients = self.list_gradients.copy()
    if self.index_mask is not None:
        train = apply_mask(list_to_mask=self.list_train,
                           mask_index=self.index_mask)[1]
        gradients = apply_mask(list_to_mask=self.list_gradients,
                               mask_index=self.index_mask)[1]

    print('Training a GP process...')
    print('Number of training points:', len(scaled_targets))

    self.gp = GaussianProcess(kernel_list=kdict,
                              regularization=0.0,
                              regularization_bounds=(0.0, 0.0),
                              train_fp=train,
                              train_target=scaled_targets,
                              gradients=gradients,
                              optimize_hyperparameters=False,
                              scale_data=False)
    self.gp.optimize_hyperparameters(global_opt=False)
    print('Optimized hyperparameters:', self.gp.theta_opt)
    print('GP process trained.')


def get_results_predicted_path(self):
    neb_tools = NEBTools(self.images)
    [self.s, self.e, self.sfit, self.efit] = neb_tools.get_fit()[0:4]
    self.path_distance = self.s[-1]
    self.uncertainty_path = []
    self.e_path = []
    for i in self.images:
        pos_unc = [i.get_positions().flatten()]
        pos_unc = apply_mask(list_to_mask=pos_unc,
                             mask_index=self.index_mask)[1]
        u = self.gp.predict(test_fp=pos_unc, uncertainty=True)
        uncertainty = 2.0 * u['uncertainty_with_reg'][0]
        i.info['uncertainty'] = uncertainty
        self.uncertainty_path.append(uncertainty)
        self.e_path.append(i.get_total_energy())
    self.images[0].info['uncertainty'] = 0.0
    self.images[-1].info['uncertainty'] = 0.0


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
            predictions = gp.predict(test_fp=test, uncertainty=False)
            return predictions['prediction'][0][0] + scaling

        Calculator.calculate(self, atoms, properties, system_changes)

        pos_flatten = self.atoms.get_positions().flatten()

        test_point = apply_mask(list_to_mask=[pos_flatten],
                                mask_index=self.ind_constraints)[1]

        # Get energy.
        energy = pred_energy_test(test=test_point)

        # Get forces:
        geom_test_pos = np.zeros((len(self.ind_constraints), len(test_point[0])))
        geom_test_neg = np.zeros((len(self.ind_constraints), len(test_point[0])))

        for i in range(len(self.ind_constraints)):
            index_force = self.ind_constraints[i]
            pos = test_point.copy()[0]
            pos[i] = pos_flatten[index_force] + self.fs

            geom_test_pos[i] = pos
            pos[i] = pos_flatten[index_force] - self.fs
            geom_test_neg[i] = pos

        f_pos = self.gp.predict(test_fp=geom_test_pos)['prediction']
        f_neg = self.gp.predict(test_fp=geom_test_neg)['prediction']

        gradients_list = (-f_neg + f_pos) / (2.0 * self.fs)
        gradients = np.zeros(len(pos_flatten))
        for i in range(len(self.ind_constraints)):
            index_force = self.ind_constraints[i]
            gradients[index_force] = gradients_list[i]

        forces = np.reshape(-gradients, (self.atoms.get_number_of_atoms(), 3))

        # Results:
        self.results['energy'] = energy
        self.results['forces'] = forces
