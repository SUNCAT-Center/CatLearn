from ase.io import Trajectory
from ase.io import read
from ase.neb import NEB
from ase.optimize import BFGS
from ase.optimize import FIRE
from ase.calculators.singlepoint import SinglePointCalculator
import ase.parallel as mpi
import numpy as np
import shutil
import os
import types
from math import log
from math import exp

class AutoNEBASE(object):
    """AutoNEB object.
    This class is a modified version of the original class AutoNEB in ASE.
    See https://gitlab.com/ase/ase/blob/master/ase/autoneb.py

    The AutoNEB algorithm streamlines the execution of NEB and CI-NEB
    calculations following the algorithm described in:

    E. L. Kolsbjerg, M. N. Groves, and B. Hammer, J. Chem. Phys,
    145, 094107, 2016. (doi: 10.1063/1.4961868)

    The user supplies at minimum the two end-points and possibly also some
    intermediate images.

    The stages are:
        1) Define a set of images and name them sequentially.
                Must at least have a relaxed starting and ending image
                User can supply intermediate guesses which do not need to
                have previously determined energies (probably from another
                NEB calculation with a lower level of theory)
        2) AutoNEB will first evaluate the user provided intermediate images
        3) AutoNEB will then add additional images dynamically until n_max
           is reached
        4) A climbing image will attempt to locate the saddle point
        5) All the images between the highest point and the starting point
           are further relaxed to smooth the path
        6) All the images between the highest point and the ending point are
           further relaxed to smooth the path

           Step 4 and 5-6 are optional steps!

    Parameters:

    attach_calculators:
        Function which adds valid calculators to the list of images supplied.
    prefix: string
        All files that the AutoNEB method reads and writes are prefixed with
        this string
    n_simul: int
        The number of relaxations run in parallel.
    n_max: int
        The number of images along the NEB path when done.
        This number includes the two end-points.
        Important: due to the dynamic adding of images around the peak n_max
        must be updated if the NEB is restarted.
    climb: boolean
        Should a CI-NEB calculation be done at the top-point
    fmax: float or list of floats
        The maximum force along the NEB path
    maxsteps: int
        The maximum number of steps in each NEB relaxation.
        If a list is given the first number of steps is used in the build-up
        and final scan phase;
        the second number of steps is used in the CI step after all images
        have been inserted.
    k: float
        The spring constant along the NEB path
    method: str (see neb.py)
        Choice between three method:
        'aseneb', standard ase NEB implementation
        'improvedtangent', published NEB implementation
        'eb', full spring force implementation (default)
    optimizer: str
        Which optimizer to use in the relaxation. Valid values are 'BFGS'
        and 'FIRE' (default)
    space_energy_ratio: float
        The preference for new images to be added in a big energy gab
        with a preference around the peak or in the biggest geometric gab.
        A space_energy_ratio set to 1 will only consider geometric gabs
        while one set to 0 will result in only images for energy
        resolution.

    The AutoNEB method uses a fixed file-naming convention.
    The initial images should have the naming prefix000.traj, prefix001.traj,
    ... up until the final image in prefix00N.traj
    Images are dynamically added in between the first and last image until
    n_max images have been reached.
    When doing the i'th NEB optimization a set of files
    prefixXXXiter00i.traj exists with XXX ranging from 000 to the N images
    currently in the NEB.

    The most recent NEB path can always be monitored by:
        $ ase-gui -n -1 neb???.traj
    """

    def __init__(self, attach_calculators, prefix, n_simul, n_max,
                 iter_folder='AutoNEB_iter',
                 fmax=0.025, maxsteps=10000, k=0.1, climb=True, method='eb',
                 optimizer='FIRE',
                 remove_rotation_and_translation=False, space_energy_ratio=0.5,
                 world=None,
                 parallel=True, smooth_curve=False, interpolate_method='idpp',
                 mic=False
                 ):
        self.mic = mic
        self.attach_calculators = attach_calculators
        self.prefix = prefix
        self.n_simul = n_simul
        self.n_max = n_max
        self.climb = climb
        self.all_images = []

        self.parallel = parallel
        self.maxsteps = maxsteps
        self.fmax = fmax
        self.k = k
        self.method = method
        self.remove_rotation_and_translation = remove_rotation_and_translation
        self.space_energy_ratio = space_energy_ratio

        if interpolate_method not in ['idpp', 'linear']:
            self.interpolate_method = 'idpp'
            print('Interpolation method not implemented.',
                  'Using the IDPP method.')
        else:
            self.interpolate_method = interpolate_method
        if world is None:
            world = mpi.world
        self.world = world
        self.smooth_curve = smooth_curve

        if optimizer == 'BFGS':
            self.optimizer = BFGS
        elif optimizer == 'FIRE':
            self.optimizer = FIRE
        else:
            raise Exception('Optimizer needs to be BFGS or FIRE')
        self.iter_folder = iter_folder
        if not os.path.exists(self.iter_folder) and self.world.rank == 0:
            os.makedirs(self.iter_folder)

    def execute_one_neb(self, n_cur, to_run, climb=False, many_steps=False):
        # Internal method which executes one NEB optimization.
        self.iteration += 1
        # First we copy around all the images we are not using in this
        # neb (for reproducibility purposes)
        if self.world.rank == 0:
            for i in range(n_cur):
                if i not in to_run[1: -1]:
                    filename = '%s%03d.traj' % (self.prefix, i)
                    t = Trajectory(filename, mode='w',
                                   atoms=self.all_images[i])
                    t.write()
                    filename_ref = self.iter_folder + \
                        '/%s%03diter%03d.traj' % (self.prefix, i,
                                                  self.iteration)
                    if os.path.isfile(filename):
                        shutil.copy2(filename, filename_ref)
        if self.world.rank == 0:
            print('Now starting iteration %d on ' % self.iteration, to_run)
        # Attach calculators to all the images we will include in the NEB

        for i in to_run[1:-1]:
            self.all_images[i].set_calculator(self.attach_calculators)

        neb = NEB([self.all_images[i] for i in to_run],
                  k=[self.k[i] for i in to_run[0:-1]],
                  method=self.method,
                  parallel=self.parallel,
                  remove_rotation_and_translation=self
                  .remove_rotation_and_translation,
                  climb=climb)

        # Do the actual NEB calculation
        qn = self.optimizer(neb,
                            logfile=self.iter_folder +
                            '/%s_log_iter%03d.log' % (self.prefix,
                                                      self.iteration))

        # Find the ranks which are masters for each their calculation
        if self.parallel:
            nneb = to_run[0]
            nim = len(to_run) - 2
            n = self.world.size // nim      # number of cpu's per image
            j = 1 + self.world.rank // n    # my image number
            assert nim * n == self.world.size
            traj = Trajectory('%s%03d.traj' % (self.prefix, j + nneb), 'w',
                              self.all_images[j + nneb],
                              master=(self.world.rank % n == 0))
            filename_ref = self.iter_folder + \
                '/%s%03diter%03d.traj' % (self.prefix,
                                          j + nneb, self.iteration)
            trajhist = Trajectory(filename_ref, 'w',
                                  self.all_images[j + nneb],
                                  master=(self.world.rank % n == 0))
            qn.attach(traj)
            qn.attach(trajhist)
        else:
            num = 1
            for i, j in enumerate(to_run[1: -1]):
                filename_ref = self.iter_folder + \
                    '/%s%03diter%03d.traj' % (self.prefix, j, self.iteration)
                trajhist = Trajectory(filename_ref, 'w', self.all_images[j])
                qn.attach(seriel_writer(trajhist, i, num).write)

                traj = Trajectory('%s%03d.traj' % (self.prefix, j), 'w',
                                  self.all_images[j])
                qn.attach(seriel_writer(traj, i, num).write)
                num += 1

        if isinstance(self.maxsteps, (list, tuple)) and many_steps:
            steps = self.maxsteps[1]
        elif isinstance(self.maxsteps, (list, tuple)) and not many_steps:
            steps = self.maxsteps[0]
        else:
            steps = self.maxsteps

        if isinstance(self.fmax, (list, tuple)) and many_steps:
            fmax = self.fmax[1]
        elif isinstance(self.fmax, (list, tuple)) and not many_steps:
            fmax = self.fmax[0]
        else:
            fmax = self.fmax
        qn.run(fmax=fmax, steps=steps)

        # Remove the calculators and replace them with single
        # point calculators and update all the nodes for
        # preparation for next iteration
        neb.distribute = types.MethodType(store_E_and_F_in_spc, neb)
        neb.distribute()

    def run(self):
        # Run the AutoNEB optimization algorithm.
        n_cur = self.__initialize__()
        while len(self.all_images) < self.n_simul + 2:
            if isinstance(self.k, (float, int)):
                self.k = [self.k] * (len(self.all_images) - 1)
            if self.world.rank == 0:
                print('Now adding images for initial run')
            # Insert a new image where the distance between two images is
            # the largest
            spring_lengths = []
            for j in range(n_cur - 1):
                spring_vec = self.all_images[j + 1].get_positions() - \
                    self.all_images[j].get_positions()
                spring_lengths.append(np.linalg.norm(spring_vec))
            jmax = np.argmax(spring_lengths)

            if self.world.rank == 0:
                print('Max length between images is at ', jmax)

            # The interpolation used to make initial guesses
            # If only start and end images supplied make all img at ones
            if len(self.all_images) == 2:
                n_between = self.n_simul
            else:
                n_between = 1

            toInterpolate = [self.all_images[jmax]]
            for i in range(n_between):
                toInterpolate += [toInterpolate[0].copy()]
            toInterpolate += [self.all_images[jmax + 1]]

            neb = NEB(toInterpolate)
            neb.interpolate(method=self.interpolate_method, mic=self.mic)

            tmp = self.all_images[:jmax + 1]
            tmp += toInterpolate[1:-1]
            tmp.extend(self.all_images[jmax + 1:])

            self.all_images = tmp

            # Expect springs to be in equilibrium
            k_tmp = self.k[:jmax]
            k_tmp += [self.k[jmax] * (n_between + 1)] * (n_between + 1)
            k_tmp.extend(self.k[jmax + 1:])
            self.k = k_tmp

            # Run the NEB calculation with the new image included
            n_cur += n_between

        # Determine if any images do not have a valid energy yet
        energies = self.get_energies()

        n_non_valid_energies = len([e for e in energies if e != e])

        if self.world.rank == 0:
            print('Start of evaluation of the initial images')

        while n_non_valid_energies != 0:
            if isinstance(self.k, (float, int)):
                self.k = [self.k] * (len(self.all_images) - 1)

            # First do one run since some energie are non-determined
            to_run, climb_safe = self.which_images_to_run_on()
            self.execute_one_neb(n_cur, to_run, climb=False)

            energies = self.get_energies()
            n_non_valid_energies = len([e for e in energies if e != e])

        if self.world.rank == 0:
            print('Finished initialisation phase.')

        # Then add one image at a time until we have n_max images
        while n_cur < self.n_max:
            if isinstance(self.k, (float, int)):
                self.k = [self.k] * (len(self.all_images) - 1)
            # Insert a new image where the distance between two images
            # is the largest OR where a higher energy resolution is needed
            if self.world.rank == 0:
                print('****Now adding another image until n_max is reached',
                      '({0}/{1})****'.format(n_cur, self.n_max))
            spring_lengths = []
            for j in range(n_cur - 1):
                spring_vec = self.all_images[j + 1].get_positions() - \
                    self.all_images[j].get_positions()
                spring_lengths.append(np.linalg.norm(spring_vec))

            total_vec = self.all_images[0].get_positions() - \
                self.all_images[-1].get_positions()
            tl = np.linalg.norm(total_vec)

            fR = max(spring_lengths) / tl

            e = self.get_energies()
            ed = []
            emin = min(e)
            enorm = max(e) - emin
            for j in range(n_cur - 1):
                delta_E = (e[j + 1] - e[j]) * (e[j + 1] + e[j] - 2 *
                                               emin) / 2 / enorm
                ed.append(abs(delta_E))

            gR = max(ed) / enorm

            if fR / gR > self.space_energy_ratio:
                jmax = np.argmax(spring_lengths)
                t = 'spring length!'
            else:
                jmax = np.argmax(ed)
                t = 'energy difference between neighbours!'

            if self.world.rank == 0:
                print('Adding image between {0} and'.format(jmax),
                      '{0}. New image point is selected'.format(jmax + 1),
                      'on the basis of the biggest ' + t)

            toInterpolate = [self.all_images[jmax]]
            toInterpolate += [toInterpolate[0].copy()]
            toInterpolate += [self.all_images[jmax + 1]]

            neb = NEB(toInterpolate)
            neb.interpolate(method=self.interpolate_method)

            tmp = self.all_images[:jmax + 1]
            tmp += toInterpolate[1:-1]
            tmp.extend(self.all_images[jmax + 1:])

            self.all_images = tmp

            # Expect springs to be in equilibrium
            k_tmp = self.k[:jmax]
            k_tmp += [self.k[jmax] * 2] * 2
            k_tmp.extend(self.k[jmax + 1:])
            self.k = k_tmp

            # Run the NEB calculation with the new image included
            n_cur += 1
            to_run, climb_safe = self.which_images_to_run_on()

            self.execute_one_neb(n_cur, to_run, climb=False)

        if self.world.rank == 0:
            print('n_max images has been reached')

        # Do a single climb around the top-point if requested
        if self.climb:
            if isinstance(self.k, (float, int)):
                self.k = [self.k] * (len(self.all_images) - 1)
            if self.world.rank == 0:
                print('****Now doing the CI-NEB calculation****')
            to_run, climb_safe = self.which_images_to_run_on()

            assert climb_safe, 'climb_safe should be true at this point!'
            self.execute_one_neb(n_cur, to_run, climb=True, many_steps=True)

        if not self.smooth_curve:
            return self.all_images

        # If a smooth_curve is requsted ajust the springs to follow two
        # gaussian distributions
        e = self.get_energies()
        peak = self.get_highest_energy_index()
        k_max = 10

        d1 = np.linalg.norm(self.all_images[peak].get_positions() -
                            self.all_images[0].get_positions())
        d2 = np.linalg.norm(self.all_images[peak].get_positions() -
                            self.all_images[-1].get_positions())
        l1 = -d1 ** 2 / log(0.2)
        l2 = -d2 ** 2 / log(0.2)

        x1 = []
        x2 = []
        for i in range(peak):
            v = (self.all_images[i].get_positions() +
                 self.all_images[i + 1].get_positions()) / 2 - \
                self.all_images[0].get_positions()
            x1.append(np.linalg.norm(v))

        for i in range(peak, len(self.all_images) - 1):
            v = (self.all_images[i].get_positions() +
                 self.all_images[i + 1].get_positions()) / 2 - \
                self.all_images[0].get_positions()
            x2.append(np.linalg.norm(v))
        k_tmp = []
        for x in x1:
            k_tmp.append(k_max * exp(-((x - d1) ** 2) / l1))
        for x in x2:
            k_tmp.append(k_max * exp(-((x - d1) ** 2) / l2))

        self.k = k_tmp
        # Roll back to start from the top-point
        if self.world.rank == 0:
            print('Now moving from top to start')
        highest_energy_index = self.get_highest_energy_index()
        nneb = highest_energy_index - self.n_simul - 1
        while nneb >= 0:
            self.execute_one_neb(n_cur, range(nneb, nneb + self.n_simul + 2),
                                 climb=False)
            nneb -= 1

        # Roll forward from the top-point until the end
        nneb = self.get_highest_energy_index()

        if self.world.rank == 0:
            print('Now moving from top to end')
        while nneb <= self.n_max - self.n_simul - 2:
            self.execute_one_neb(n_cur, range(nneb, nneb + self.n_simul + 2),
                                 climb=False)
            nneb += 1
        return self.all_images

    def __initialize__(self):
        # Load files from the filesystem.
        if not os.path.isfile('%s000.traj' % self.prefix):
            raise IOError('No file with name %s000.traj' % self.prefix,
                          'was found. Should contain initial image')

        # Find the images that exist
        index_exists = [i for i in range(self.n_max) if
                        os.path.isfile('%s%03d.traj' % (self.prefix, i))]

        n_cur = index_exists[-1] + 1

        if self.world.rank == 0:
            print('The NEB initially has %d images ' % len(index_exists),
                  '(including the end-points)')
        if len(index_exists) == 1:
            raise Exception('Only a start point exists')

        for i in range(len(index_exists)):
            if i != index_exists[i]:
                raise Exception('Files must be ordered sequentially',
                                'without gaps.')
        if self.world.rank == 0:
            for i in index_exists:
                filename_ref = self.iter_folder + \
                    '/%s%03diter000.traj' % (self.prefix, i)
                if os.path.isfile(filename_ref):
                    try:
                        os.rename(filename_ref, filename_ref + '.bak')
                    except IOError:
                        pass
                filename = '%s%03d.traj' % (self.prefix, i)
                try:
                    shutil.copy2(filename, filename_ref)
                except IOError:
                    pass
        # Wait for file system on all nodes is synchronized
        self.world.barrier()
        # And now lets read in the configurations
        for i in range(n_cur):
            if i in index_exists:
                filename = '%s%03d.traj' % (self.prefix, i)
                newim = read(filename)
                self.all_images.append(newim)
            else:
                self.all_images.append(self.all_images[0].copy())

        self.iteration = 0
        return n_cur

    def get_energies(self):
        """Utility method to extract all energies and insert np.NaN at
        invalid images."""
        energies = []
        for a in self.all_images:
            try:
                energies.append(a.get_potential_energy())
            except RuntimeError:
                energies.append(np.NaN)
        return energies

    def get_energies_one_image(self, image):
        """Utility method to extract energy of an image and return np.NaN
        if invalid."""
        try:
            energy = image.get_potential_energy()
        except RuntimeError:
            energy = np.NaN
        return energy

    def get_highest_energy_index(self):
        """Find the index of the image with the highest energy."""
        energies = self.get_energies()
        valid_entries = [(i, e) for i, e in enumerate(energies) if e == e]
        highest_energy_index = max(valid_entries, key=lambda x: x[1])[0]
        return highest_energy_index

    def which_images_to_run_on(self):
        """Determine which set of images to do a NEB at.
        The priority is to first include all images without valid energies,
        secondly include the highest energy image."""
        n_cur = len(self.all_images)
        energies = self.get_energies()
        # Find out which image is the first one missing the energy and
        # which is the last one missing the energy
        first_missing = n_cur
        last_missing = 0
        n_missing = 0
        for i in range(1, n_cur - 1):
            if energies[i] != energies[i]:
                n_missing += 1
                first_missing = min(first_missing, i)
                last_missing = max(last_missing, i)

        highest_energy_index = self.get_highest_energy_index()

        nneb = highest_energy_index - 1 - self.n_simul // 2
        nneb = max(nneb, 0)
        nneb = min(nneb, n_cur - self.n_simul - 2)
        nneb = min(nneb, first_missing - 1)
        nneb = max(nneb + self.n_simul, last_missing) - self.n_simul
        to_use = range(nneb, nneb + self.n_simul + 2)

        while self.get_energies_one_image(self.all_images[to_use[0]]) != \
                self.get_energies_one_image(self.all_images[to_use[0]]):
            to_use[0] -= 1
        while self.get_energies_one_image(self.all_images[to_use[-1]]) != \
                self.get_energies_one_image(self.all_images[to_use[-1]]):
            to_use[-1] += 1

        return to_use, (highest_energy_index in to_use[1: -1])


class seriel_writer:
    def __init__(self, traj, i, num):
        self.traj = traj
        self.i = i
        self.num = num

    def write(self):
        if self.num % (self.i + 1) == 0:
            self.traj.write()


def store_E_and_F_in_spc(self):
    """Collect the energies and forces on all nodes and store as
    single point calculators"""
    # Make sure energies and forces are known on all nodes
    self.get_forces()
    images = self.images
    if self.parallel:
        energy = np.empty(1)
        forces = np.empty((self.natoms, 3))

        for i in range(1, self.nimages - 1):
            # Determine which node is the leading for image i
            root = (i - 1) * self.world.size // (self.nimages - 2)
            # If on this node, extract the calculated numbers
            if self.world.rank == root:
                energy[0] = images[i].get_potential_energy()
                forces = images[i].get_forces()
            # Distribute these numbers to other nodes
            self.world.broadcast(energy, root)
            self.world.broadcast(forces, root)
            # On all nodes, remove the calculator, keep only energy
            # and force in single point calculator
            self.images[i].set_calculator(
                SinglePointCalculator(self.images[i],
                                      energy=energy[0],
                                      forces=forces))
