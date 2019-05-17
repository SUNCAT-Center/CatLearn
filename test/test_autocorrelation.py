from catlearn.featurize.setup import FeatureGenerator
from catlearn.utilities.neighborlist import ase_connectivity
from ase.build import molecule
from ase.data import covalent_radii
import numpy as np
import unittest


truth = np.array([[166, 15963, 39.8497, 220, 27890,
                   61.444, 172, 21422, 65.1592]])


class TestAutoCorrelation(unittest.TestCase):
    """Test the autocorrelation feature generator."""

    def test_generator(self):
        """Test the feature generation."""
        atoms = molecule('HCOOH')
        atoms.center(vacuum=5)
        radii = [covalent_radii[z] + 0.1 for z in atoms.numbers]
        atoms.connectivity = ase_connectivity(atoms, radii)
        images = [atoms]
        gen = FeatureGenerator()
        features = gen.return_vec(images, [gen.get_autocorrelation])

        np.testing.assert_allclose(features, truth)


if __name__ == '__main__':
    unittest.main()
