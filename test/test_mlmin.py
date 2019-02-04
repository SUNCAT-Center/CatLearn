import unittest
import numpy as np
from catlearn.optimize.mlmin import MLMin
from ase import Atoms
from ase.io import read, write
from ase.calculators.emt import EMT
from ase.build import fcc100, add_adsorbate

# Set up initial geometry:
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)
write('initial_mlmin.traj', slab)


class TestMLMin(unittest.TestCase):
    """ General test of the ML-Min algorithm."""
    def test_minimize(self):
        initial_structure = read('initial_mlmin.traj')
        initial_structure.set_calculator(EMT())

        initial_opt = MLMin(initial_structure,
                            trajectory='mlmin_structures.traj')
        initial_opt.run(fmax=0.01, full_output=True)

        atoms_catlearn = read('mlmin_structures.traj', ':')
        n_eval_catlearn = len(atoms_catlearn)
        print('Checking number of function calls...')
        self.assertEqual(n_eval_catlearn, 6)
        print('Checking converged energy...')
        e_opt = initial_opt.list_targets[-1]
        e_test = 3.310696
        np.testing.assert_array_almost_equal(e_opt, e_test, decimal=5)
        print('Checking converged fmax...')
        fmax_opt = initial_opt.list_fmax[-1][0]
        fmax_test = 0.006456
        np.testing.assert_array_almost_equal(fmax_opt, fmax_test, decimal=5)

    def test_acquisition(self):
        """Test ML-Min acquisition functions"""

        # Test acquisition lcb:
        initial_structure = read('initial_mlmin.traj')
        initial_structure.set_calculator(EMT())

        initial_opt = MLMin(initial_structure,
                            trajectory='mlmin_structures.traj')
        initial_opt.run(fmax=0.01, full_output=True, acq='lcb')

        atoms_catlearn = read('mlmin_structures.traj', ':')
        n_eval_catlearn = len(atoms_catlearn)
        print('Checking number of function calls...')
        self.assertEqual(n_eval_catlearn, 7)
        print('Checking converged energy...')
        e_opt = initial_opt.list_targets[-1]
        e_test = 3.310696
        np.testing.assert_array_almost_equal(e_opt, e_test, decimal=5)
        print('Checking converged fmax...')
        fmax_opt = initial_opt.list_fmax[-1][0]
        fmax_test = 0.00837
        np.testing.assert_array_almost_equal(fmax_opt, fmax_test, decimal=5)

        # Test acquisition ucb:
        initial_structure = read('initial_mlmin.traj')
        initial_structure.set_calculator(EMT())

        initial_opt = MLMin(initial_structure,
                            trajectory='mlmin_structures.traj')
        initial_opt.run(fmax=0.01, full_output=True, acq='ucb')

        atoms_catlearn = read('mlmin_structures.traj', ':')
        n_eval_catlearn = len(atoms_catlearn)
        print('Checking number of function calls...')
        self.assertEqual(n_eval_catlearn, 6)
        print('Checking converged energy...')
        e_opt = initial_opt.list_targets[-1]
        e_test = 3.310696
        np.testing.assert_array_almost_equal(e_opt, e_test, decimal=5)
        print('Checking converged fmax...')
        fmax_opt = initial_opt.list_fmax[-1][0]
        fmax_test = 0.008297
        np.testing.assert_array_almost_equal(fmax_opt, fmax_test, decimal=5)

    def test_kernel(self):
        """Test ML-Min kernels"""

        # Test kernel ARD fixed:
        initial_structure = read('initial_mlmin.traj')
        initial_structure.set_calculator(EMT())

        initial_opt = MLMin(initial_structure,
                            trajectory='mlmin_structures.traj')
        initial_opt.run(fmax=0.01, full_output=True, acq='min_energy',
                        kernel='SQE_fixed')

        atoms_catlearn = read('mlmin_structures.traj', ':')
        n_eval_catlearn = len(atoms_catlearn)
        print('Checking number of function calls...')
        self.assertEqual(n_eval_catlearn, 6)
        print('Checking converged energy...')
        e_opt = initial_opt.list_targets[-1]
        e_test = 3.310719
        np.testing.assert_array_almost_equal(e_opt, e_test, decimal=5)
        print('Checking converged fmax...')
        fmax_opt = initial_opt.list_fmax[-1][0]
        fmax_test = 0.008573
        np.testing.assert_array_almost_equal(fmax_opt, fmax_test, decimal=5)

        # Test ARD SQE kernel:
        initial_structure = read('initial_mlmin.traj')
        initial_structure.set_calculator(EMT())

        initial_opt = MLMin(initial_structure,
                            trajectory='mlmin_structures.traj')
        initial_opt.run(fmax=0.01, full_output=True, acq='min_energy',
                        kernel='ARD_SQE')

        atoms_catlearn = read('mlmin_structures.traj', ':')
        n_eval_catlearn = len(atoms_catlearn)
        print('Checking number of function calls...')
        self.assertEqual(n_eval_catlearn, 6)
        print('Checking converged energy...')
        e_opt = initial_opt.list_targets[-1]
        e_test = 3.310702
        np.testing.assert_array_almost_equal(e_opt, e_test, decimal=5)
        print('Checking converged fmax...')
        fmax_opt = initial_opt.list_fmax[-1][0]
        fmax_test = 0.006957
        np.testing.assert_array_almost_equal(fmax_opt, fmax_test, decimal=5)

