from ase.io import read
from ase.neb import NEB
import numpy as np
from catlearn.optimize.mlneb import MLNEB
import copy
from catlearn.optimize.functions_calc import MullerBrown
from ase import Atoms
from ase.optimize import BFGS
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
    """ General test of the ML-NEB algorithm."""
    def test_path(self):
        """Test ML-NEB algorithm running with an interpolated path"""
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
                             interpolation=images,
                             ase_calc=ase_calculator,
                             restart=False
                             )

        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj', max_step=0.2,
                         full_output=True)

        atoms_catlearn = read('evaluated_structures.traj', ':')
        n_eval_catlearn = len(atoms_catlearn) - 2

        self.assertEqual(n_eval_catlearn, 13)
        print('Checking number of function calls using 8 images...')
        np.testing.assert_array_equal(n_eval_catlearn, 13)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.0468
        print('Checking uncertainty on the path (8 images):')
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

    def test_restart(self):
        """ Here we test the restart flag, the mic, and the internal
            interpolation."""

        # Checking internal interpolation.

        neb_catlearn = MLNEB(start='initial_optimized.traj',
                             end='final_optimized.traj',
                             n_images=9,
                             ase_calc=ase_calculator,
                             interpolation='linear',
                             restart=False
                             )
        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj', max_step=0.2)
        print('Checking number of iterations using 9 images...')
        self.assertEqual(neb_catlearn.iter, 12)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.0377
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
        neb_catlearn.run(fmax=0.01, max_step=0.20,
                         unc_convergence=0.010,
                         trajectory='ML-NEB.traj')
        print('Checking number of iterations restarting with 11 images...')
        self.assertEqual(neb_catlearn.iter, 5)
        print('Checking uncertainty on the path (11 images).')
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.0062
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

        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj', max_step=0.2,
                         acquisition='acq_1')
        self.assertEqual(neb_catlearn.iter, 16)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.0028
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        print('Checking acquisition function 2 using 6 images...')
        neb_catlearn = MLNEB(start='initial_optimized.traj',
                             end='final_optimized.traj',
                             n_images=6,
                             ase_calc=ase_calculator,
                             restart=False
                             )
        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj', max_step=0.2,
                         acquisition='acq_2')

        self.assertEqual(neb_catlearn.iter, 13)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.0128
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        print('Checking acquisition function 3 using 6 images...')
        neb_catlearn = MLNEB(start='initial_optimized.traj',
                             end='final_optimized.traj',
                             n_images=6,
                             ase_calc=ase_calculator,
                             restart=False
                             )
        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj', max_step=0.2,
                         acquisition='acq_3')

        self.assertEqual(neb_catlearn.iter, 14)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.0036
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        print('Checking acquisition function 4 using 6 images...')
        neb_catlearn = MLNEB(start='initial_optimized.traj',
                             end='final_optimized.traj',
                             n_images=6,
                             ase_calc=ase_calculator,
                             restart=False
                             )
        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj', max_step=0.2,
                         acquisition='acq_4')

        self.assertEqual(neb_catlearn.iter, 16)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.0028
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

        print('Checking acquisition function 5 using 6 images...')
        neb_catlearn = MLNEB(start='initial_optimized.traj',
                             end='final_optimized.traj',
                             n_images=6,
                             ase_calc=ase_calculator,
                             restart=False
                             )
        neb_catlearn.run(fmax=0.05, trajectory='ML-NEB.traj', max_step=0.2,
                         acquisition='acq_5')

        self.assertEqual(neb_catlearn.iter, 13)
        max_unc = np.max(neb_catlearn.uncertainty_path)
        unc_test = 0.0128
        np.testing.assert_array_almost_equal(max_unc, unc_test, decimal=4)

if __name__ == '__main__':
    unittest.main()
