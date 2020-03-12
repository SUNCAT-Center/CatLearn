import numpy as np
from catlearn.optimize.io import ase_to_catlearn, store_results_neb, \
                                 print_version, store_trajectory_neb, \
                                 print_info_neb, array_to_ase, print_cite_mlneb
from catlearn.optimize.constraints import create_mask, apply_mask
from ase.neb import NEB
from ase.neb import NEBTools
from ase.io import read, write
from ase.optimize import MDMin
from ase.parallel import parprint, rank, parallel_function
from scipy.spatial import distance
import os
from catlearn.regression import GaussianProcess
from ase.calculators.calculator import Calculator, all_changes
from ase.atoms import Atoms
from catlearn import __version__


class MLNEB(object):

    def __init__(self, start, end, prev_calculations=None,
                 n_images=0.25, k=None, interpolation='linear', mic=False,
                 neb_method='improvedtangent', ase_calc=None, restart=True,
                 force_consistent=None):

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
        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K). By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.

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

        # Start end-point, final end-point and path (optional).
        self.start = start
        self.end = end
        self.n_images = n_images
        self.feval = 0

        # General setup.
        self.fc = force_consistent
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

        msg = 'Error: Initial structure for the NEB was not provided.'
        assert start is not None, msg
        msg = 'Error: Final structure for the NEB was not provided.'
        assert end is not None, msg
        msg = 'ASE calculator not provided (see "ase_calc" flag).'
        assert self.ase_calc, msg

        is_endpoint = read(start, '-1:')
        fs_endpoint = read(end, '-1:')
        is_pos = is_endpoint[-1].get_positions().flatten()
        fs_pos = fs_endpoint[-1].get_positions().flatten()

        # Check the magnetic moments of the initial and final states:
        self.magmom_is = is_endpoint[-1].get_initial_magnetic_moments()
        self.magmom_fs = fs_endpoint[-1].get_initial_magnetic_moments()

        # Convert atoms information into data to feed the ML process.

        # Include Restart mode and previous calculations.

        if restart is not True:
            merged_trajectory = is_endpoint + fs_endpoint
            trj = ase_to_catlearn(merged_trajectory)
            write('./evaluated_structures.traj', is_endpoint + fs_endpoint)

        if restart is True or prev_calculations is not None:
            if prev_calculations is None:
                eval_file = 'evaluated_structures.traj'
            if prev_calculations is not None:
                eval_file = prev_calculations
            if os.path.exists(eval_file):
                eval_atoms = read(eval_file, ':')
                trj = ase_to_catlearn(eval_atoms)
            if not os.path.exists(eval_file):
                merged_trajectory = is_endpoint + fs_endpoint
                trj = ase_to_catlearn(merged_trajectory)
                write('./evaluated_structures.traj', is_endpoint + fs_endpoint)

        self.list_train, self.list_targets, self.list_gradients, trj_images, \
            self.constraints, self.num_atoms = [trj['list_train'],
                                                trj['list_targets'],
                                                trj['list_gradients'],
                                                trj['images'],
                                                trj['constraints'],
                                                trj['num_atoms']]
        self.ase_ini = read(start)
        self.num_atoms = len(self.ase_ini)
        if len(self.constraints) < 0:
            self.constraints = None
        if self.constraints is not None:
            self.index_mask = create_mask(self.ase_ini, self.constraints)

        # Obtain the energy of the endpoints for scaling:
        self.energy_is = is_endpoint[-1].get_potential_energy(
                                                      force_consistent=self.fc)
        self.energy_fs = fs_endpoint[-1].get_potential_energy(
                                                      force_consistent=self.fc)

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
        self.path_distance = self.d_start_end.copy()

        # Get forces for the previous steps
        self.list_max_abs_forces = []
        for i in self.list_gradients:
                self.list_fmax = get_fmax(np.array([i]))
                self.max_abs_forces = np.max(np.abs(self.list_fmax))
                self.list_max_abs_forces.append(self.max_abs_forces)

        print_info_neb(self)

    def run(self, fmax=0.05, unc_convergence=0.050, steps=500,
            trajectory='ML_NEB_catlearn.traj', acquisition='acq_5',
            dt=0.025, ml_steps=750, max_step=0.25, sequential=False,
            full_output=False):

        """Executing run will start the NEB optimization process.

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
        ml_steps: int
            Maximum number of steps for the NEB optimization on the
            predicted landscape.
        max_step: float
            Early stopping criteria. Maximum uncertainty before stopping the
            optimization in the predicted landscape.
        sequential: boolean
            When sequential is set to True, the ML-NEB algorithm starts
            with only one moving image. After finding a saddle point
            the algorithm adds all the images selected in the MLNEB class
            (the total number of NEB images is defined in the 'n_images' flag).
        full_output: boolean
            Whether to print on screen the full output (True) or not (False).

        Returns
        -------
        Minimum Energy Path from the initial to the final states.

        """
        self.acq = acquisition
        self.fullout = full_output

        # Calculate a third point if only known initial & final structures.
        if len(self.list_targets) == 2:
            middle = int(self.n_images * (2./3.))
            if self.energy_is >= self.energy_fs:
                middle = int(self.n_images * (1./3.))
            self.interesting_point = \
                self.images[middle].get_positions().flatten()

            eval_and_append(self, self.interesting_point)

            self.iter += 1
            self.max_forces = get_fmax(np.array([self.list_gradients[-1]]))
            self.max_abs_forces = np.max(np.abs(self.max_forces))
            self.list_max_abs_forces.append(self.max_abs_forces)
            print_info_neb(self)

            store_trajectory_neb(self)

        stationary_point_found = False

        org_n_images = self.n_images

        if sequential is True:
            self.n_images = 3

        while True:

            # 1. Train Machine Learning process.
            self.gp, self.max_target = \
                train_gp_model(self.list_train, self.list_targets,
                               self.list_gradients, self.index_mask,
                               self.path_distance, self.fullout)

            # 2. Setup and run ML NEB:
            if self.fullout is True:
                parprint('Max number steps:', ml_steps)
            ml_cycles = 0

            while True:

                if stationary_point_found is True:
                    self.n_images = org_n_images

                starting_path = self.images  # Start from last path.

                if ml_cycles == 0:
                    sp = '0:' + str(self.n_images)
                    if self.fullout is True:
                        parprint('Using initial path.')
                    starting_path = read('./all_predicted_paths.traj', sp)

                if ml_cycles == 1:
                    if self.fullout is True:
                        parprint('Using last predicted path.')
                    sp = str(-self.n_images) + ':'
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

                # Test before optimization:

                for i in self.images:
                    i.get_potential_energy()
                    get_results_predicted_path(self)
                    unc_ml = np.max(self.uncertainty_path[1:-1])

                if unc_ml >= max_step:
                    if self.fullout is True:
                        parprint('Maximum uncertainty reach in initial path.')
                        parprint('Early stop.')
                    break

                # Perform NEB in the predicted landscape.
                ml_neb = NEB(self.images, climb=True,
                             method=self.neb_method,
                             k=self.spring)
                if self.fullout is True:
                    parprint('Optimizing ML CI-NEB using dt:', dt)
                neb_opt = MDMin(ml_neb, dt=dt, logfile=None)
                if full_output is True:
                    neb_opt = MDMin(ml_neb, dt=dt)

                ml_converged = False
                n_steps_performed = 0
                while ml_converged is False:
                    # Save prev. positions:
                    prev_save_positions = []

                    for i in self.images:
                        prev_save_positions.append(i.get_positions())

                    neb_opt.run(fmax=(fmax * 0.85), steps=1)
                    neb_opt.nsteps = 0

                    n_steps_performed += 1
                    get_results_predicted_path(self)
                    unc_ml = np.max(self.uncertainty_path[1:-1])
                    e_ml = np.max(self.e_path[1:-1])

                    if e_ml >= self.max_target + 0.2:
                        for i in range(0, self.n_images):
                            self.images[i].positions = prev_save_positions[i]
                        if self.fullout is True:
                            parprint('Pred. energy above max. energy. '
                                     'Early stop.')
                        ml_converged = True

                    if unc_ml >= max_step:
                        for i in range(0, self.n_images):
                            self.images[i].positions = prev_save_positions[i]
                        if self.fullout is True:
                            parprint('Maximum uncertainty reach. Early stop.')
                        ml_converged = True
                    if neb_opt.converged():
                        ml_converged = True

                    if np.isnan(ml_neb.emax):
                        sp = str(-self.n_images) + ':'
                        self.images = read('./all_predicted_paths.traj', sp)
                        for i in self.images:
                            i.get_potential_energy()
                        n_steps_performed = 10000

                    if n_steps_performed > ml_steps-1:
                        if self.fullout is True:
                            parprint('Not converged yet...')
                        ml_converged = True

                if n_steps_performed <= ml_steps-1:
                    if self.fullout is True:
                        parprint('Converged opt. in the predicted landscape.')
                    break

                ml_cycles += 1
                if self.fullout is True:
                    parprint('ML cycles performed:', ml_cycles)

                if ml_cycles == 2:
                    if self.fullout is True:
                        parprint('ML process not optimized...not safe...')
                        parprint('Change interpolation or numb. of images.')
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
                                     self.argmax_unc].get_positions().flatten()

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
                    # If stationary point is found behave like acq. 2.
                    if stationary_point_found is True:
                        # Select image with max. uncertainty.
                        self.argmax_unc = \
                                     np.argmax(self.uncertainty_path[1:-1])
                        self.interesting_point = self.images[1:-1][
                                 self.argmax_unc].get_positions().flatten()

                    # Select image with max. predicted value.
                    if np.max(self.uncertainty_path[1:-1]) < \
                                                           unc_convergence:

                        self.argmax_unc = np.argmax(pred_plus_unc)
                        self.interesting_point = \
                            self.images[1:-1][int(
                                self.argmax_unc)].get_positions().flatten()

            # 5. Add a new training point and evaluate it.
            if self.fullout is True:
                parprint('Performing evaluation on the real landscape...')
            eval_and_append(self, self.interesting_point)
            self.iter += 1
            if self.fullout is True:
                parprint('Single-point calculation finished.')

            # 6. Store results.
            parprint('\n')
            self.energy_forward = np.max(self.e_path) - self.e_path[0]
            self.energy_backward = np.max(self.e_path) - self.e_path[-1]
            self.max_forces = get_fmax(np.array([self.list_gradients[-1]]))
            self.max_abs_forces = np.max(np.abs(self.max_forces))

            print_info_neb(self)
            store_results_neb(self)
            store_trajectory_neb(self)

            # 7. Check convergence:

            if self.max_abs_forces <= fmax:
                stationary_point_found = True

            # Check whether the evaluated point is a stationary point.
            if self.max_abs_forces <= fmax and self.n_images == org_n_images:
                msg = "Congratulations! Stationary point is found! "
                msg2 = "Check the file 'evaluated_structures.traj' using ASE."
                parprint(msg+msg2)

                if np.max(self.uncertainty_path[1:-1]) < unc_convergence:
                    # Save results of the final step (converged):
                    self.gp, self.max_target = \
                        train_gp_model(self.list_train, self.list_targets,
                                       self.list_gradients, self.index_mask,
                                       self.path_distance, self.fullout)
                    get_results_predicted_path(self)
                    store_results_neb(self)
                    msg = "Congratulations! Your ML NEB is converged. "
                    msg2 = "If you want to plot the ML NEB predicted path you "
                    msg3 = "should check the files 'results_neb.csv' "
                    msg4 = "and 'results_neb_interpolation.csv'."
                    parprint(msg+msg2+msg3+msg4)
                    # Last path.
                    write(trajectory, self.images)
                    parprint('The optimized predicted path can be found in: ',
                             trajectory)
                    # Clean up:
                    if rank == 0:
                        os.remove('./last_predicted_path.traj')
                        os.remove('./all_predicted_paths.traj')
                    break

            # Break if reaches the max number of iterations set by the user.
            if steps <= self.iter:
                parprint('Maximum number iterations reached. Not converged.')
                break

        parprint('Number of steps performed in total:',
                 len(self.list_targets)-2)
        print_cite_mlneb()


def create_ml_neb(is_endpoint, fs_endpoint, images_interpolation,
                  n_images, constraints, index_constraints,
                  scaling_targets, iteration, gp=None):
    """
    Generates input NEB for the GPR.
    """

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


@parallel_function
def train_gp_model(list_train, list_targets, list_gradients, index_mask,
                   path_distance, fullout=False):
    """
    Train Gaussian process
    """
    max_target = np.max(list_targets)
    scaled_targets = list_targets.copy() - max_target
    sigma_f = 1e-3 + np.std(scaled_targets)**2

    dimension = 'single'
    bounds = ((0.1, path_distance),)

    width = path_distance / 2

    if np.isnan(width) or width <= 0.05:
        width = path_distance / 2

    noise_energy = 0.005
    noise_forces = 0.0005

    kdict = [{'type': 'gaussian', 'width': width,
              'dimension': dimension,
              'bounds': bounds,
              'scaling': sigma_f,
              'scaling_bounds': ((sigma_f, sigma_f),)},
             {'type': 'noise_multi',
              'hyperparameters': [noise_energy, noise_forces],
              'bounds': ((0.001, 0.005),
                         (0.0005, 0.002),)}
             ]

    train = list_train.copy()
    gradients = list_gradients.copy()
    if index_mask is not None:
        train = apply_mask(list_to_mask=list_train,
                           mask_index=index_mask)[1]
        gradients = apply_mask(list_to_mask=list_gradients,
                               mask_index=index_mask)[1]
    parprint('\n')
    parprint('Training a Gaussian process...')
    parprint('Number of training points:', len(scaled_targets))

    gp = GaussianProcess(kernel_list=kdict,
                         regularization=0.0,
                         regularization_bounds=(0.0, 0.0),
                         train_fp=train,
                         train_target=scaled_targets,
                         gradients=gradients,
                         optimize_hyperparameters=False,
                         scale_data=False)
    gp.optimize_hyperparameters(global_opt=False)
    if fullout:
        parprint('Optimized hyperparameters:', gp.kernel_list)
    parprint('Gaussian process trained.')

    return gp, max_target

def get_results_predicted_path(self):

    """
    Obtain results from the predicted NEB.
    """

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
        geom_test_pos = np.zeros((len(self.ind_constraints),
                                  len(test_point[0])))
        geom_test_neg = np.zeros((len(self.ind_constraints),
                                  len(test_point[0])))

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


def get_fmax(gradients_flatten):

    """
    Function that print a list of max. individual atom forces.
    """

    forces_flatten = -gradients_flatten
    list_fmax = np.zeros((len(gradients_flatten), 1))
    j = 0
    for i in forces_flatten:
        atoms_forces_i = np.reshape(i, (-1, 3))
        list_fmax[j] = np.max(np.sqrt(np.sum(atoms_forces_i**2, axis=1)))
        j = j + 1
    return list_fmax


def get_energy_catlearn(self, x=None):

    """ Evaluates the objective function at a given point in space.

    Parameters
    ----------
    self: arrays
        Previous information from the CatLearn optimizer.
    x : array
        Array containing the atomic positions (flatten).

    Returns
    -------
    energy : float
        The function evaluation value.
    """
    energy = 0.0

    # If no point is passed, evaluate the last trained point.
    if x is None:
        x = self.list_train[-1]

    # Get energies using ASE:
    pos_ase = array_to_ase(x, self.num_atoms)

    self.ase_ini.set_calculator(None)
    self.ase_ini = Atoms(self.ase_ini, positions=pos_ase,
                         calculator=self.ase_calc)
    energy = self.ase_ini.get_potential_energy(force_consistent=self.fc)
    return energy


def get_forces_catlearn(self, x=None):

    """ Evaluates the forces (ASE) or the Jacobian of the objective
    function at a given point in space.

    Parameters
    ----------
    self: arrays
        Previous information from the CatLearn optimizer.
    x : array
        Atoms positions or point in space.

    Returns
    -------
    forces : array
        Forces of the atomic structure (flatten).
    """
    forces = 0.0
    # If no point is passed, evaluate the last trained point.
    if x is None:
        x = self.list_train[-1]

    # Get energies using ASE:
    forces = self.ase_ini.get_forces().flatten()
    return forces


def eval_and_append(self, interesting_point):
    """ Evaluates the energy and forces (ASE) of the point of interest
        for a given atomistic structure.

    Parameters
    ----------
    self: arrays
        Previous information from the CatLearn optimizer.
    interesting_point : ndarray
        Atoms positions or point in space.

    Return
    -------
    Append function evaluation and forces values to the training set.
    """

    if np.ndim(interesting_point) == 1:
        interesting_point = np.array([interesting_point])

    self.list_train = np.append(self.list_train,
                                interesting_point, axis=0)
    
    # Remove old calculation information 
    self.ase_calc.results = {}
    
    energy = get_energy_catlearn(self)

    self.list_targets = np.append(self.list_targets, energy)

    gradients = [-get_forces_catlearn(self).flatten()]
    self.list_gradients = np.append(self.list_gradients,
                                    gradients, axis=0)

    self.list_targets = np.reshape(self.list_targets,
                                   (len(self.list_targets), 1))

    self.feval += 1
