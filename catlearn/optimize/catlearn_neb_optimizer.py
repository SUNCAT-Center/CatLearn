import numpy as np
import re
from catlearn.optimize.gp_calculator import GPCalculator
from catlearn.optimize.warnings import *
from catlearn.optimize.constraints import *
from catlearn.optimize.convergence import *
from catlearn.optimize.initialize import *
from catlearn.optimize.neb_tools import *
from ase import Atoms
from scipy.spatial import distance
from catlearn.optimize.catlearn_ase_calc import CatLearn_ASE
from ase.io import read, write
# from ase.visualize import view
from ase.neb import NEBTools
import copy
from catlearn.optimize.plots import plot_neb_mullerbrown, plot_predicted_neb_path
from ase.optimize import QuasiNewton, BFGS, FIRE, MDMin

class NEBOptimizer(object):

    def __init__(self, start=None, end=None, path=None, ml_calc=None, \
    ase_calc=None,
    acq_fun=None, filename='results', inc_prev_calcs=False, n_images=None,
    interpolation='linear'):
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
        acq_fun : string
            Acquisition function (UCB, EI, PI).
            Default is None (Max uncertainty of the NEB predicted path.)
        filename: string
            Filename to store the output.
        """

        # Start end-point, final end-point and path (optional) :
        self.start = start
        self.end = end
        self.path = path

        # General setup:
        self.ml_calc = ml_calc
        self.ase_calc = ase_calc
        self.acq_fun = acq_fun
        self.filename = filename
        self.ase = True
        self.inc_prev_calcs = inc_prev_calcs

        # Reset:
        self.constraints = None
        self.interesting_point = None

        self.ml_calc = ml_calc
        if self.ml_calc is None:
            self.kdict = {'k1': {'type': 'gaussian', 'scaling': 1.0, 'width':
                          0.5, 'dimension':'single'}}
            self.ml_calc = GPCalculator(
            kernel_dict=self.kdict, opt_hyperparam=False, scale_data=False,
            scale_optimizer=False,
            calc_uncertainty=False, regularization=1e-5)

        assert start is not None, err_not_neb_start()
        assert end is not None, err_not_neb_end()

        self.x0 = start
        self.f0 = end
        self.ase = True
        self.ase_calc = ase_calc
        assert self.ase_calc, err_not_ase_calc_traj()

        # A) Include previous calculations for training the ML model.
        if self.inc_prev_calcs is True:
            self.trajectory_start = read(self.x0, ':')
            self.trajectory_end = read(self.f0, ':')

        # B) Only include initial and final (optimized) images.
        if self.inc_prev_calcs is False:
            self.trajectory_start = read(self.x0,'-1:')
            self.trajectory_end = read(self.f0, '-1:')
        self.posx0 = (self.trajectory_start[-1].get_positions().flatten())
        self.posf0 = (self.trajectory_end[-1].get_positions().flatten())

        # Convert atoms information to ML information.
        if os.path.exists('./tmp.traj'):
                os.remove('./tmp.traj')
        merged_trajectory = self.trajectory_start + self.trajectory_end
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
        if len(self.constraints) < 0:
            self.constraints = None
        if self.constraints is not None:
            self.ind_mask_constr = create_mask_ase_constraints(
            self.ase_ini, self.constraints)

        # Calculate length of the path.
        self.d_start_end = distance.euclidean(self.posx0,self.posf0)

        # Initial path can be provided or interpolated from the end-points.

        # A) ASE Interpolation.
        if self.path is None:
            self.n_images = n_images
            self.interpolation = interpolation
            initial_images = initialize_neb(self)

        # B) User provides a path.
        if self.path is not None:
            #! One must assert that it is a traj file here.
            print('User has provided a path.')
            path_images = read(self.path, ':')

            # Check if provided full path or only the intermediate images:
            start_positions = read(self.x0).get_positions().flatten()
            final_positions = read(self.f0).get_positions().flatten()
            path_positions_is = path_images[0].get_positions().flatten()
            path_positions_fs = path_images[-1].get_positions().flatten()
            if np.sum(start_positions-path_positions_is) == 0:
                print('The initial state was already in path.')
                path_images = path_images[1:] # Remove first image.
            if np.sum(final_positions-path_positions_fs) == 0:
                print('The final state was already in path.')
                path_images = path_images[:-1] # Remove last image.
            self.path_images = path_images
            initial_images = initialize_neb(self)

        # Save initial path in list of paths.
        write('all_pred_paths.traj', initial_images[:])

        # Attach labels to the end-points.
        label_start_prev = read(self.start, ':')
        label_start_prev[-1].info['label'] = 0
        write(self.start, label_start_prev)
        label_end_prev = read(self.end, ':')
        label_end_prev[-1].info['label'] = self.n_images
        write(self.end, label_end_prev)


    def run(self, fmax=0.05, ml_fmax=None, unc_conv=0.025, max_iter=200,
            max_step=None, climb_img=False, neb_method='improvedtangent',
            ml_algo='FIRE', penalty_mode='all_neb_atoms_label_v2', k=None,
            store_neb_paths=False):

        """Executing run will start the optimization process.

        Parameters
        ----------
        ml_algo : string
            Algorithm for the NEB optimization of the predicted mean or
            acquisition function. Implemented:
            'FIRE_ASE' and 'MDMin_ASE'.
        neb_method: string
            Implemented are: 'aseneb', 'improvedtangent' and 'eb'.
            (ASE function).
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html#module-ase.neb
        climb_img: bool
            Use a climbing image.
            (ASE function).
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html#module-ase.neb
        k: float or list of floats.
            Spring constant(s) in eV/Ang. One number or one for each spring.
            (ASE function).
            See https://wiki.fysik.dtu.dk/ase/ase/neb.html#module-ase.neb
        max_step: float
            Maximum step size in Ang. before applying a penalty (atoms or
            distances depending on the penalty mode).
        penalty_mode: string
            Penalization mode.
        fmax: float
            Convergence criteria. Forces below fmax.
        store_neb_paths: bool
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
        self.fmax = fmax

        if ml_fmax is None:
            ml_fmax = fmax
        self.initial_ml_fmax = ml_fmax
        self.ml_fmax = ml_fmax
        self.unc_conv = unc_conv

        self.iter = 0
        self.max_iter = max_iter
        self.ml_algo = ml_algo

        self.climb_img = climb_img
        self.ci = False

        self.neb_method = neb_method
        self.initial_k = k
        self.k = k

        self.penalty_mode = penalty_mode
        self.max_step = max_step


        print('Number of images:', self.n_images)
        print('Distance between first and last point:', self.d_start_end)
        print('Max step:', self.max_step)
        print('Spring constant:', self.k)

        self.iter = 0

        while not neb_converged(self):

            self.org_train = self.list_train.copy()
            self.org_targets = self.list_targets.copy()
            self.org_gradients = self.list_gradients.copy()

            if self.constraints is not None:
                [self.org_train, self.list_train] = apply_mask_ase_constraints(
                                           list_to_mask=self.list_train,
                                           mask_index=self.ind_mask_constr)
                [self.org_gradients, self.list_gradients] = \
                                           apply_mask_ase_constraints(
                                           list_to_mask=self.list_gradients,
                                           mask_index= self.ind_mask_constr)

            # Scale energies:
            mean_targets = np.mean(self.list_targets)
            self.list_targets = self.list_targets - mean_targets

            self.trained_process = self.ml_calc.train_process(
                    train_data=self.list_train,
                    target_data=self.list_targets,
                    gradients_data=self.list_gradients)

            if self.ml_calc.__dict__['opt_hyperparam']:
                self.ml_calc.opt_hyperparameters()

            # 2) Optimize ML and return the next point to evaluate:
            self.ml_fmax = self.initial_ml_fmax*2
            self.max_ml_steps = 100 # Hard-coded.
            if self.ci is True:
                self.ml_fmax = self.initial_ml_fmax
                self.max_ml_steps = 150 # Hard-coded.

            ml_algo_i = re.sub('\_ASE$', '', self.ml_algo)
            warning_ml_algo(self)
            warning_climb_img(self)

            start_guess_ml = read(self.x0)
            final_guess_ml = read(self.f0)

            self.images = [start_guess_ml]

            # Scale energies (initial):
            self.images[0].__dict__['_calc'].__dict__['results']['energy'] = \
            self.images[0].__dict__['_calc'].__dict__['results']['energy'] - \
            mean_targets

            # Append images:

            last_images = read("all_pred_paths.traj", '-' + str(self.n_images)
             + ':')

            for i in range(1, self.n_images-1):
                image = start_guess_ml.copy()
                image.info['label'] = i
                image.info['uncertainty'] = 0
                image.set_calculator(CatLearn_ASE(
                                     trained_process=self.trained_process,
                                     ml_calc=self.ml_calc,
                                     finite_step=1e-5,
                                     max_step=self.max_step,
                                     n_images=self.n_images,
                                     penalty_mode=self.penalty_mode,
                                     c_crit_penalty=1.0)
                                     )
                image.set_positions(last_images[i].get_positions())
                image.set_constraint(self.constraints)
                self.images.append(image)

            # Scale energies (final):
            self.images.append(final_guess_ml)
            self.images[-1].__dict__['_calc'].__dict__['results']['energy'] = \
            self.images[-1].__dict__['_calc'].__dict__['results']['energy'] - \
                mean_targets

            # Set up ML+NEB:
            neb = NEB(self.images, climb=self.ci,
                      method=self.neb_method, k=self.k)  # Hard-coded.

            # Set up ML+NEB:
            if ml_algo is 'FIRE' or ml_algo is 'MDMin':
                neb_opt = eval(ml_algo_i)(neb, dt=0.1)
            if ml_algo is not 'FIRE' or ml_algo is not 'MDMin':
                neb_opt = eval(ml_algo_i)(neb)

            # Run ML+NEB:
            neb_opt.run(fmax=self.ml_fmax, steps=self.max_ml_steps)


            # Get dist. convergence (d between opt. path and prev. opt. path).

            self.list_distance_convergence = []
            for i in range(0, len(self.images)): # Only mid positions
                pos_opt_i = self.images[i].get_positions().flatten()
                pos_last_i = last_images[i].get_positions().flatten()
                self.list_distance_convergence.append(distance.sqeuclidean(
                                                 pos_opt_i, pos_last_i))
            self.distance_convergence = np.max(
            np.abs(self.list_distance_convergence))
            print('Max. distance between last path and current path',
                  self.distance_convergence)

            # Get energies and uncertainties of the predicted path.

            self.ml_calc.__dict__['calc_uncertainty']= True

            self.energies_discr_neb = []
            self.unc_discr_neb = []
            energy_img0 = self.images[0].get_total_energy()
            for i in self.images:
                energy_i = i.get_total_energy()
                pos_i = [i.get_positions().flatten()]
                pos_i_masked = apply_mask_ase_constraints(
                                 list_to_mask=pos_i,
                                 mask_index=self.ind_mask_constr)[1]
                pred_i = self.ml_calc.get_predictions(
                                    trained_process=self.trained_process,
                                    test_data=pos_i_masked[0])
                print(pred_i)
                unc_i = pred_i['uncertainty_with_reg']
                i.info['uncertainty']= unc_i[0]
                #! CHECK THIS (why different?):
                #! print(energy_i)
                #! print(pred_i)
                #! CHECK THIS:
                self.unc_discr_neb.append(unc_i)
                self.energies_discr_neb.append(energy_i - energy_img0)

            self.energies_discr_neb = np.asarray(self.energies_discr_neb).flatten()
            self.unc_discr_neb = np.asarray(self.unc_discr_neb).flatten()
            self.unc_discr_neb[0] = 0.0
            self.unc_discr_neb[-1] = 0.0
            self.ml_calc.__dict__['calc_uncertainty']= False

            # Save (append) opt. path (including initial and final images).

            all_prev_paths = read("all_pred_paths.traj",':') + self.images
            write('all_pred_paths.traj', all_prev_paths)

            # Get fit of the discrete path.
            neb_tools = NEBTools(self.images)
            [s, E, Sfit, Efit, lines] = neb_tools.get_fit()
            if store_neb_paths is True:
                plot_predicted_neb_path(images=self.images, iter=self.iter)

            # Select image with max. uncertainty.
            self.interesting_point = self.images[np.argmax(
            self.unc_discr_neb)].get_positions().flatten()

            # Check if the path found is reliable.
            ml_conv = True

            if neb_opt.__dict__['nsteps'] >= self.max_ml_steps:
                print('Max. numb. ML iterations exceeded.')
                ml_conv = False

            if self.distance_convergence >= 0.20:
                print('The path was too stretched. The last NEB is not saved.')
                ml_conv = False
                # Accept the path if the uncertainty goes below 50 meV
                if np.max(self.unc_discr_neb) <= (2*unc_conv):
                    ml_conv = True

            # If it is not reliable remove last positions for penalty.
            # This is when the path is stretch too much or when NEB ML did
            # not converged.

            if ml_conv is False:
                all_prev_paths = read("all_pred_paths.traj", ':'+'-' + str(self.n_images))
                write('all_pred_paths.traj', all_prev_paths)

            #############################################
            if store_neb_paths is True:
                if self.ase_calc.__dict__['name'] == 'mullerbrown':
                    plot_neb_mullerbrown(self)
            #############################################

            # 3) Add a new training point and evaluate it.

            self.list_train = self.org_train.copy()
            self.list_targets = self.org_targets.copy() # Rescale the targets
            self.list_gradients = self.org_gradients.copy()
            if self.interesting_point.ndim == 1:
                self.interesting_point = np.array([self.interesting_point])
            self.list_train = np.append(self.list_train,
                                        self.interesting_point, axis=0)
            self.list_targets = np.append(self.list_targets,
                                          get_energy_catlearn(self))
            self.list_targets = np.reshape(self.list_targets,
                                          (len(self.list_targets), 1))
            self.list_gradients = np.append(self.list_gradients,
                [-get_forces_catlearn(self).flatten()], axis=0)

            # Store all the atoms objects of the real evaluations (expensive).

            if self.iter == 0:
                TrajectoryWriter(atoms=self.ase_ini, filename='./' + str(self.filename)
                     +'_evaluated_images.traj', mode='w').write()
            if self.iter > 0:
                TrajectoryWriter(atoms=self.ase_ini, filename='./' + str(self.filename)
                     +'_evaluated_images.traj', mode='a').write()

            # Break if reaches the max number of iterations set by the user.
            if self.max_iter <= self.iter:
                warning_max_iter_reached()
                break

            self.iter += 1

            self.feval = len(self.list_targets-2)

            print('Length of initial path:', self.d_start_end)
            print('Length discrete path:', s[-1])
            print('Max Step:', self.max_step)
            print('Max uncertainty:', np.max(self.unc_discr_neb))
            print('NEB ML Converged?', ml_conv)
            print('ITERATIONS:', self.iter)
            print('Spring constant:', self.k)

