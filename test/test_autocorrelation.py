from catlearn.fingerprint.molecule_fingerprint import AutoCorrelationFingerprintGenerator as ACG
from ase.build import molecule
import numpy as np
import unittest

truth = np.array([[166.0, 15963.0, 39.849700000000006,
                   232.0, 32416.0, 72.664, 178.0, 22910.0, 78.7552]])


class TestAutoCorrelation(unittest.TestCase):
    """Test the autocorrelation feature generator."""

    def test_generator(self):
        """Test the feature generation."""
        atoms = molecule('HCOOH')
        gen = ACG(atoms, dstar=2)
        features = gen.generate()

        np.testing.assert_allclose(features, truth)


if __name__ == '__main__':
    unittest.main()
