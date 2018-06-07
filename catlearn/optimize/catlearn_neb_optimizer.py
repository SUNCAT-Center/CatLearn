import numpy as np
import re
from catlearn.optimize.gp_calculator import GPCalculator
from catlearn.optimize.warnings import *
from catlearn.optimize.constraints import *
from catlearn.optimize.convergence import *
from catlearn.optimize.initialize import *
from catlearn.optimize.neb_tools import *
from catlearn.optimize.neb_functions import *
from ase import Atoms
from scipy.spatial import distance
from ase.io import read, write
# from ase.visualize import view
from ase.neb import NEBTools
import copy
from catlearn.optimize.plots import plot_neb_mullerbrown, plot_predicted_neb_path
from ase.optimize import QuasiNewton, BFGS, FIRE, MDMin


class NEBOptimizer(object):

    def __init__(self, start=None, end=None, path=None,
                 ml_calc=None, ase_calc=None, filename='results',
                 inc_prev_calcs=False, n_images=None,
                 interpolation='',
                 remove_rotation_and_translation=False, mic=False,
                 climb_image=True):
        """ Nudged elastic band (NEB) setup.

        Parameters
        ----------
        start: Trajectory file or atoms object (ASE)
            First end point of the NEB path. Starting geometry.
        end: Trajectory file or atoms object (ASE)
            Second end point of the NEB path. Final geometry.
        path: Trajectory file or atoms object (ASE)
            Atoms trajectory with the intermediate images of the path
            between the two end-points.
        ml_calc : ML calculator object
            Machine Learning calculator (e.g. Gaussian Processes).
            Default is GP.
        ase_calc: ASE calculator object
            When using ASE the user must pass an ASE calculator.
            Default is None (Max uncertainty of the NEB predicted path.)
        filename: string
            Filename to store the output.
        n_images: number of images including initial and final images (
        end-points).
        """

        # Start end-point, final end-point and path (optional):
        self.start = start
        self.end = end
        self.path = path
        self.n_images = n_images

        # General setup:
        self.iter = 0
        self.ml_calc = ml_calc
        self.ase_calc = ase_calc
        self.filename = filename
        self.ase = True

        # Reset:
        self.constraints = None
        self.interesting_point = None

        assert start is not None, err_not_neb_start()
        assert end is not None, err_not_neb_end()

        self.ase = True
        self.ase_calc = ase_calc
        assert self.ase_calc, err_not_ase_calc_traj()

        # A) Include previous calculations for training the ML model.
        is_end_point = read(start, ':')
        fs_end_point = read(end, ':')

        # Write previous evaluated images in evaluations:
        write('./' + str(self.filename) +'_evaluated_images.traj',
              is_end_point + fs_end_point)

        # B) Only include initial and final (optimized) images.
        if inc_prev_calcs is False:
            is_end_point = read(start, '-1:')
            fs_end_point = read(end, '-1:')
        is_pos = is_end_point[-1].get_positions().flatten()
        fs_pos = fs_end_point[-1].get_positions().flatten()



        # Check the magnetic moments of the initial and final states:
        self.magmom_is = None
        self.magmom_fs = None
        self.magmom_is = is_end_point[-1].get_initial_magnetic_moments()
        self.magmom_fs = fs_end_point[-1].get_initial_magnetic_moments()

        self.spin = False
        if not np.array_equal(self.magmom_is, np.zeros_like(self.magmom_is)):
            self.spin = True
        if not np.array_equal(self.magmom_fs, np.zeros_like(self.magmom_fs)):
            self.spin = True


        # Obtain the energy of the endpoints for scaling:
        energy_is = is_end_point[-1].get_potential_energy()
        energy_fs = fs_end_point[-1].get_potential_energy()
        self.scale_targets = np.min([energy_is, energy_fs])

        # Convert atoms information to ML information.
        if os.path.exists('./tmp.traj'):
                os.remove('./tmp.traj')
        merged_trajectory = is_end_point + fs_end_point
        write('tmp.traj', merged_trajectory)

        trj = ase_traj_to_catlearn(traj_file='tmp.traj',
                                ase_calc=self.ase_calc)
        self.list_train, self.list_targets, self.list_gradients, \
            trj_images, self.constraints, self.num_atoms = [
            trj['list_train'], trj['list_targets'],
            trj['list_gradients'], trj['images'],
            trj['constraints'], trj['num_atoms']]
        os.remove('./tmp.traj')
        self.ase_ini =  trj_images[0]
        self.num_atoms = len(self.ase_ini)
        if len(self.constraints) < 0:
            self.constraints = None
        if self.constraints is not None:
            self.ind_mask_constr = create_mask_ase_constraints(
            self.ase_ini, self.constraints)

        # Calculate length of the path.
        self.d_start_end = distance.euclidean(is_pos, fs_pos)

        # Configure ML.

        if self.ml_calc is None:
            self.kdict = {
                         'k1': {'type': 'gaussian', 'width': 0.5,
                         'bounds':
                          ((0.1, 1.0),)*len(self.ind_mask_constr),
                         'scaling': 1.0, 'scaling_bounds': ((1.0, 1.0),)},
                        }
            self.ml_calc = GPCalculator(
            kernel_dict=self.kdict, opt_hyperparam=False, scale_data=False,
            scale_optimizer=False,
            calc_uncertainty=False, regularization=1e-5)

        # Settings of the NEB.
        self.rrt = remove_rotation_and_translation
        self.mic = mic
        self.climb_image = climb_image
        self.ci = False

        self.neb_dict = {'max_step': 0.10,
                         'a_const': 100.0,
                         'c_const': 10.0,
                         'scale_targets': self.scale_targets,
                         'iteration': self.iter,
                         'constraints': self.constraints,
                         'ind_constraints': self.ind_mask_constr,
                         'method': 'improvedtangent',
                         'spring_k': 100.0,
                         'n_images': self.n_images,
                         'initial_endpoint': is_end_point[-1],
                         'final_endpoint': fs_end_point[-1],
                         'all_pred_images': [],
                         'last_accepted_images': []}

        # Initial path can be provided or interpolated from the end-points.

        # Attach labels to the end-points.

        # if self.path is None:
        #
        #     label_start_prev = read(self.start, ':')
        #     label_start_prev[-1].info['label'] = 0
        #     write(self.start, label_start_prev)
        #     label_end_prev = read(self.end, ':')
        #     label_end_prev[-1].info['label'] = self.n_images
        #     write(self.end, label_end_prev)

        # A) ASE NEB Interpolation.
        if self.path is None:
            self.images = create_ml_neb(images_interpolation=None,
                                        ml_calculator=self.ml_calc,
                                        trained_process=None,
                                        settings_neb_dict=self.neb_dict)

            neb = NEB(self.images, remove_rotation_and_translation=self.rrt)
            neb.interpolate(method=interpolation, mic=self.mic)


        # B) User provides a path.
        # if self.path is not None:
        #     images_path = read(path, ':') # images
        #
        #     if not np.array_equal(images_path[0].get_positions.flatten(),
        #                           is_pos):
        #         images_path.insert(0, is_end_point[-1])
        #     if not np.array_equal(images_path[-1].get_positions.flatten(),
        #                           fs_pos):
        #         images_path.append(fs_end_point[-1])
        #
        #     self.neb_dict['n_images'] = len(images_path)
        #     self.images = create_ml_neb(images_interpolation=images_path,
        #                   ml_calculator=self.ml_calc,
        #                   trained_process=None,
        #                   settings_neb_dict=self.neb_dict,
        #                   iteration=self.iter
        #                   )

        # Save initial path in list of paths.


        # Tag images as accepted:
        for i in self.images:
            i.info['accepted_path'] = True
        self.neb_dict['all_pred_images'] = self.images
        self.neb_dict['last_accepted_images'] = self.images

        # Initial path as accepted path:
        write('last_accepted_path.traj', self.images)


        # Check whether the user set enough images.
        assert self.n_images > 3, err_not_enough_images()

        # Start dictionary for run.
        self.converg_dict = {}


    def run(self, fmax=0.05, max_iter=30, unc_conv=0.025, ml_algo='FIRE',
            plot_neb_paths=False):

        """Executing run will start the optimization process.

        Parameters
        ----------
        ml_algo : string
            Algorithm for the surrogate model:
            'FIRE_ASE' and 'MDMin_ASE'.
        neb_method: string
            Implemented are: 'aseneb', 'improvedtangent' and 'eb'.
            (ASE function).
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html#module-ase.neb
        climb_img: bool
            Use a climbing image.
            (ASE function).
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html#module-ase.neb
        max_step: float
            Maximum step size in Ang. before applying a penalty (atoms or
            distances depending on the penalty mode).
        fmax: float
            Convergence criteria. Forces below fmax.
        plot_neb_paths: bool
            If True it stores the predicted NEB path in pdf format for each
            iteration.
            Note: Python package matplotlib is required.

        Returns
        -------
        The NEB results are printed in filename.txt (results.txt, if the
        user don't specify a name).
        A trajectory file in ASE format (filename.traj)
        containing the geometries of the system for each step
        is also generated. A dictionary with a summary of the results can be
        accessed by self.results and it is also printed in 'filename'_data.txt.

        """

        self.converg_dict = {'fmax': fmax,
                                 'uncertainty_convergence': unc_conv,
                                 'max_iter': max_iter,
                                 'ml_fmax': fmax * 2.0,
                                 'ml_max_iter': 500}

        print('Number of images:', self.neb_dict['n_images'])
        print('Distance between first and last point:', self.d_start_end)
        print('Max step:', self.neb_dict['max_step'])
        print('Spring constant:', self.neb_dict['spring_k'])


        # while not neb_converged(self):
        while True:

            # 1) Train Machine Learning process:

            # Check that you are not feeding redundant information.
            count_unique = np.unique(self.list_train, return_counts=True,
                                     axis=0)[1]
            msg = 'Your training list constains 1 or more duplicated elements'
            assert not np.any(count_unique>1), msg

            process = train_ml_process(list_train=self.list_train,
                                       list_targets=self.list_targets,
                                       list_gradients=self.list_gradients,
                                       index_constraints=self.ind_mask_constr,
                                       ml_calculator=self.ml_calc,
                                       scale_targets=self.scale_targets)
            
            trained_process = process['trained_process']
            ml_calc = process['ml_calc']


            # 2) Setup and run ML NEB:

            # Start from last accepted path:

            images_guess = self.neb_dict['last_accepted_images']


            # Create images of the path:
            self.images = create_ml_neb(
                                    images_interpolation=images_guess,
                                    trained_process=trained_process,
                                    ml_calculator=ml_calc,
                                    settings_neb_dict=self.neb_dict)
            warning_climb_img(self.ci)

            neb = NEB(self.images, climb=False,
                      method=self.neb_dict['method'],
                      k=self.neb_dict['spring_k'],
                      remove_rotation_and_translation=self.rrt)


            neb_opt = eval(ml_algo)(neb, dt=0.1)

            neb_opt.run(fmax=self.converg_dict['ml_fmax'],
                        steps=self.converg_dict['ml_max_iter'])


            # # Improve the previous ML NEB path by using CI-NEB:
            self.ci = False
            if self.iter > 0 and np.max(uncertainty_path) <= \
                                self.converg_dict['uncertainty_convergence']:
                if self.climb_image is True:
                    warning_climb_img(self.climb_image)
                    self.neb_dict['a_const'] = 1.0
                    self.neb_dict['c_const'] = 1.0
                    images_ci = copy.deepcopy(self.images)
                    neb_ci = NEB(images_ci, climb=True,
                                 method='improvedtangent',
                                 k=100.0,
                                 remove_rotation_and_translation=self.rrt)
                    neb_opt_ci = eval(ml_algo)(neb_ci, dt=0.1)
                    neb_opt_ci.run(fmax=self.converg_dict['fmax'],
                            steps=self.converg_dict['ml_max_iter'])
                    conv_ci = neb_opt_ci.__dict__['nsteps'] < \
                              self.converg_dict['ml_max_iter']
                    # Accept the CI optimization if it is converged:
                    if conv_ci is True:
                        self.ci = True
                        self.images = copy.deepcopy(images_ci)
                        neb_opt = neb_opt_ci

            # 3) Get results from ML NEB:

            # Store the path obtain after the run in dictionary:
            self.neb_dict['all_pred_images'] += self.images

            # Get fit of the discrete path.
            neb_tools = NEBTools(self.images)
            [s, E, Sfit, Efit, lines] = neb_tools.get_fit()

            # Get dist. convergence (d between opt. path and prev. opt. path).

            list_distance_convergence = []
            list_distance_acceptance = []
            for i in range(0, len(self.images)):
                pos_opt_i = self.images[i].get_positions().flatten()
                pos_last_i = self.neb_dict['last_accepted_images'][
                                           i].get_positions().flatten()
                list_distance_convergence.append(distance.euclidean(
                                                 pos_opt_i, pos_last_i))
                list_distance_acceptance.append(distance.sqeuclidean(
                                                 pos_opt_i, pos_last_i))


            distance_convergence = np.max(np.abs(
                                               list_distance_convergence))
            distance_acceptance = np.max(np.abs(
                                               list_distance_acceptance))
            print('Max. distance between last accepted path and current path',
                  distance_convergence)

            # Tag uncertainty to the images of the path.

            ml_calc.__dict__['calc_uncertainty']= True
            for i in range(1, len(self.images)-1): # Only to the middle images.
                pos_i = [self.images[i].get_positions().flatten()]
                pos_i_masked = apply_mask_ase_constraints(
                                 list_to_mask=pos_i,
                                 mask_index=self.ind_mask_constr)[1]
                pred_i = ml_calc.get_predictions(
                                    trained_process=trained_process,
                                    test_data=pos_i_masked[0])
                unc_i = pred_i['uncertainty_with_reg']
                self.images[i].info['uncertainty']= unc_i[0]
            ml_calc.__dict__['calc_uncertainty']= False

            uncertainty_path = [i.info['uncertainty'] for i in self.images]
            energies_path = [i.get_potential_energy() for i in self.images]

            # Tag if the path found is reliable.

            # A) Not accept a path that was stretched too much.
            if distance_acceptance >= 2 * self.neb_dict['max_step']:
                print('The path was too stretched. The last NEB is not saved.')
                for i in self.images:
                    i.info['accepted_path'] = False

            # B) Not accept a path in which the ML NEB did not converged.
            if neb_opt.__dict__['nsteps'] >= self.converg_dict['ml_max_iter']:
                print('Max. numb. ML iterations exceeded.')
                for i in self.images:
                    i.info['accepted_path'] = False

            # C) Accept path if the uncertainty goes below 2*unc_conv in eV.
            if np.max(uncertainty_path) <= (2*unc_conv):
                for i in self.images:
                    i.info['accepted_path'] = True

            # Tag iteration:
            for i in self.images:
                    i.info['iteration'] = self.iter

            ##################################################################
            # Select image with max. uncertainty.
            argm_unc = np.argmax(uncertainty_path)
            interesting_point = self.images[argm_unc].get_positions().flatten()
            ##################################################################

            # Store plots.
            if plot_neb_paths is True:
                if self.ase_calc.__dict__['name'] == 'mullerbrown':
                    plot_neb_mullerbrown(images=self.images,
                                         interesting_point=interesting_point,
                                         trained_process=trained_process,
                                         list_train=self.list_train)

            if plot_neb_paths is True:
                plot_predicted_neb_path(images=self.images,
                                        climb_image=self.ci,
                                        filename=self.filename)

            # 3) Add a new training point and evaluate it.

            evaluate_interesting_point_and_append_training(self,
                                                          interesting_point)

            # Save (append) opt. path (including initial and final images).

            write('all_pred_paths.traj', self.neb_dict['all_pred_images'])
            write('last_pred_path.traj', self.images)
            if self.images[-1].info['accepted_path'] is True:
                write('last_accepted_path.traj', self.images)

            # Store all the atoms objects of the real evaluations (expensive).
            TrajectoryWriter(atoms=self.ase_ini, filename='./' + str(
                             self.filename) +'_evaluated_images.traj',
                             mode='a').write()

            print('Maximum step set by the user:', self.neb_dict['max_step'])
            print('Spring constant:', self.neb_dict['spring_k'])
            print('Length of initial path:', self.d_start_end)
            print('Length of the current path:', s[-1])
            print('Max uncertainty:', np.max(uncertainty_path))
            print('NEB ML Converged / Path accepted?:', self.images[0].info[
                                                        'accepted_path'])
            print('Number of data points trained:', len(self.list_targets))
            print('ITERATIONS:', self.iter)



            # Break if reaches the max number of iterations set by the user.
            if self.converg_dict['max_iter'] <= self.iter:
                warning_max_iter_reached()
                break

        # Print Final convergence:
        print('Number of function evaluations in this run:', self.iter)
