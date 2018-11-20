"""Test io functions."""
import unittest
import os
import numpy as np

from catlearn.regression import GaussianProcess
from catlearn.regression.gpfunctions import io
from common import get_data

wkdir = os.getcwd()


class TestIO(unittest.TestCase):

    def train_model(self, train_features, train_targets):
        """Function to train a Gaussian process."""
        kdict = [{'type': 'gaussian', 'width': 0.5, 'scaling': 1.},
                 {'type': 'linear', 'scaling': 1.},
                 {'type': 'constant', 'const': 1.},
                 {'type': 'quadratic', 'slope': 1., 'degree': 1.,
                  'scaling': 1.}]
        self.__class__.gp = GaussianProcess(
            train_fp=train_features, train_target=train_targets,
            kernel_list=kdict, regularization=np.sqrt(1e-3),
            optimize_hyperparameters=True, scale_data=False)

        io.write(filename='test-model', model=self.gp, ext='pkl')
        io.write(filename='test-model', model=self.gp, ext='hdf5')

    def make_test(self, train_features, train_targets, test_features,
                  test_targets):
        """Function to test Gaussian process."""
        self.train_model(train_features, train_targets)

        self.original = self.gp.predict(
            test_fp=test_features, test_target=test_targets,
            get_validation_error=True, get_training_error=True)

    def test_load(self):
        """Function to test loading a pre-generated model."""
        train_features, train_targets, test_features, test_targets = get_data()

        self.make_test(
            train_features, train_targets, test_features, test_targets)

        new_gp = io.read(filename='test-model', ext='pkl')

        pred = new_gp.predict(
            test_fp=test_features, test_target=test_targets,
            get_validation_error=True, get_training_error=True)

        self.assertTrue(np.allclose(
            pred['validation_error']['rmse_all'],
            self.original['validation_error']['rmse_all']))

        gp = io.read(filename='test-model', ext='hdf5')

        pred = gp.predict(test_fp=test_features,
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)

        self.assertTrue(
            np.allclose(pred['validation_error']['rmse_all'],
                        self.original['validation_error']['rmse_all']))

        os.remove('{}/test-model.pkl'.format(wkdir))
        os.remove('{}/test-model.hdf5'.format(wkdir))

    def test_raw(self):
        """Function to test raw data save."""
        regularization = self.gp.regularization
        kernel_list = self.gp.kernel_list

        train_features, train_targets, test_features, test_targets = get_data()
        io.write_train_data('train_data', train_features, train_targets,
                            regularization, kernel_list)
        tf, tt, r, klist = io.read_train_data('train_data')
        self.assertTrue(np.allclose(train_features, tf) and
                        np.allclose(train_targets, tt))
        self.assertTrue(r == regularization)
        for i, kernel in enumerate(kernel_list):
            for key in kernel:
                if type(kernel[key]) != list and \
                        type(kernel[key]) != np.ndarray:
                    self.assertTrue(klist[i][key] == kernel[key])
                else:
                    self.assertTrue(np.allclose(klist[i][key],
                                                kernel[key]))
        os.remove('{}/train_data.hdf5'.format(wkdir))


if __name__ == '__main__':
    unittest.main()
