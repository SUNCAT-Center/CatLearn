from ase.io import read
from ase.neb import NEB
import numpy as np
from catlearn.optimize.mlneb import MLNEB
import copy
from catlearn.optimize.functions_calc import MullerBrown
from ase import Atoms
from ase.optimize import BFGS
import os
import unittest

# 1. Structural relaxation.

# Setup calculator:
ase_calculator = MullerBrown()

# # 1.1. Structures:
initial_structure = Atoms('C', positions=[(-0.55, 1.41, 0.0)])
final_structure = Atoms('C', positions=[(0.626, 0.025, 0.0)])

initial_structure.set_calculator(copy.deepcopy(ase_calculator))
final_structure.set_calculator(copy.deepcopy(ase_calculator))

# 1.2. Optimize initial and final end-points.

# Initial end-point:
initial_opt = BFGS(initial_structure, trajectory='initial_optimized.traj')
initial_opt.run(fmax=0.01)

# Final end-point:
final_opt = BFGS(final_structure, trajectory='final_optimized.traj')
final_opt.run(fmax=0.01)


class TestMLNEB(unittest.TestCase):
    """ General test of the ML-NEB case. In this case we provide a path."""
    def test_path(self):
        n_images = 8
        images = [initial_structure]
        for i in range(1, n_images-1):
            image = initial_structure.copy()
            image.set_calculator(copy.deepcopy(ase_calculator))
            images.append(image)
        images.append(final_structure)

        neb = NEB(images, climb=True)
        neb.interpolate(method='linear')

        neb_catlearn = MLNEB(start=initial_structure,
                             end=final_structure,
                             path=images,
                             ase_calc=ase_calculator,
                             restart=False
                             )

        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj')

        atoms_catlearn = read('evaluated_structures.traj', ':')
        n_eval_catlearn = len(atoms_catlearn) - 2

        self.assertEqual(n_eval_catlearn, 12)
        print('Checking number of function calls using 8 images...')
        np.testing.assert_array_equal(n_eval_catlearn, 12)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.022412729039317382
        print('Checking uncertainty on the path (8 images):')
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

    def test_restart(self):
        """ Here we test the restart flag, the mic, and the internal
            interpolation."""

        # Checking internal interpolation

        neb_catlearn = MLNEB(start='initial_optimized.traj',
                             end='final_optimized.traj',
                             n_images=9,
                             ase_calc=ase_calculator
                             )
        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj')
        print('Checking number of iterations using 9 images...')
        self.assertEqual(neb_catlearn.iter, 11)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.05204091495394993
        print('Checking uncertainty on the path (9 images):')
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        # Reducing the uncertainty and fmax, varying num. images (restart):
        print("Checking restart flag...")
        print('Using tighter convergence criteria.')

        neb_catlearn = MLNEB(start='initial_optimized.traj',
                             end='final_optimized.traj',
                             n_images=11,
                             ase_calc=ase_calculator,
                             restart=True
                             )
        neb_catlearn.run(fmax=0.01,
                         unc_convergence=0.010,
                         trajectory='ML-NEB.traj')
        print('Checking number of iterations restarting with 11 images...')
        self.assertEqual(neb_catlearn.iter, 4)
        print('Checking uncertainty on the path (11 images).')
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.009596096929423526
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

    def test_acquisition(self):
        """ Here we test the acquisition functions"""

        print('Checking acquisition function 1 using 6 images...')
        neb_catlearn = MLNEB(start='initial_optimized.traj',
                             end='final_optimized.traj',
                             n_images=6,
                             ase_calc=ase_calculator,
                             restart=False
                             )

        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj',
                         acquisition='acq_1')
        self.assertEqual(neb_catlearn.iter, 12)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.016837502479194518
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        print('Checking acquisition function 2 using 6 images...')
        neb_catlearn = MLNEB(start='initial_optimized.traj',
                             end='final_optimized.traj',
                             n_images=6,
                             ase_calc=ase_calculator,
                             restart=False
                             )
        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj',
                         acquisition='acq_2')

        self.assertEqual(neb_catlearn.iter, 10)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.019377964708766612
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        print('Checking acquisition function 3 using 6 images...')
        neb_catlearn = MLNEB(start='initial_optimized.traj',
                             end='final_optimized.traj',
                             n_images=6,
                             ase_calc=ase_calculator,
                             restart=False
                             )
        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj',
                         acquisition='acq_3')

        self.assertEqual(neb_catlearn.iter, 10)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.02956129325684482
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        print('Checking acquisition function 4 using 6 images...')
        neb_catlearn = MLNEB(start='initial_optimized.traj',
                             end='final_optimized.traj',
                             n_images=6,
                             ase_calc=ase_calculator,
                             restart=False
                             )
        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj',
                         acquisition='acq_4')

        self.assertEqual(neb_catlearn.iter, 12)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.016837502479194518
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        print('Checking acquisition function 5 using 6 images...')
        neb_catlearn = MLNEB(start='initial_optimized.traj',
                             end='final_optimized.traj',
                             n_images=6,
                             ase_calc=ase_calculator,
                             restart=False
                             )
        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj',
                         acquisition='acq_5')

        self.assertEqual(neb_catlearn.iter, 10)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.019377964708766612
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        # # Cleaning:
        #
        # os.remove('all_predicted_paths.traj')
        # os.remove('evaluated_structures.traj')
        # os.remove('final.traj')
        # os.remove('final_optimized.traj')
        # os.remove('initial.traj')
        # os.remove('initial_optimized.traj')
        # os.remove('results_neb.csv')
        # os.remove('results_neb_interpolation.csv')
        # os.remove('ML-NEB.traj')
        # os.remove('warnings_and_errors.txt')


if __name__ == '__main__':
    unittest.main()
