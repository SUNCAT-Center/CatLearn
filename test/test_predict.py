"""Script to test the prediction functions."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import unittest

from catlearn.regression import RidgeRegression, GaussianProcess
from catlearn.regression.gpfunctions.sensitivity import SensitivityAnalysis
from catlearn.setup.general_gp import GeneralGaussianProcess
from catlearn.regression.cost_function import get_error

from common import get_data

train_size, test_size = 45, 5


class TestPrediction(unittest.TestCase):
    """Test out the various prediction routines."""

    def test_rr_loocv(self):
        """Test ridge regression predictions with loocv fitting."""
        train_features, train_targets, test_features, test_targets = get_data()

        # Test ridge regression predictions.
        rr = RidgeRegression(cv='loocv')
        reg = rr.find_optimal_regularization(X=train_features, Y=train_targets)
        coef = rr.RR(X=train_features, Y=train_targets, omega2=reg)[0]

        # Test the model.
        sumd = 0.
        for tf, tt in zip(test_features, test_targets):
            p = (np.dot(coef, tf))
            sumd += (p - tt) ** 2
        print('Ridge regression prediction:',
              (sumd / len(test_features)) ** 0.5)

    def test_rr_bootstrap(self):
        """Test ridge regression predictions with bootstrap fitting."""
        train_features, train_targets, test_features, test_targets = get_data()

        # Test ridge regression predictions.
        rr = RidgeRegression(cv='bootstrap')
        reg = rr.find_optimal_regularization(X=train_features, Y=train_targets)
        coef = rr.RR(X=train_features, Y=train_targets, omega2=reg)[0]

        # Test the model.
        sumd = 0.
        for tf, tt in zip(test_features, test_targets):
            p = (np.dot(coef, tf))
            sumd += (p - tt) ** 2
        print('Ridge regression prediction:',
              (sumd / len(test_features)) ** 0.5)

    def test_gp_linear_kernel(self):
        """Test Gaussian process predictions with the linear kernel."""
        train_features, train_targets, test_features, test_targets = get_data()

        # Test prediction routine with linear kernel.
        kdict = {'k1': {'type': 'linear', 'scaling': 1.,
                        'scaling_bounds': ((0., None),)},
                 'c1': {'type': 'constant', 'const': 1.}}
        gp = GaussianProcess(
            train_fp=train_features, train_target=train_targets,
            kernel_dict=kdict, regularization=1e-3,
            optimize_hyperparameters=True, scale_data=True)
        pred = gp.predict(test_fp=test_features,
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)
        self.assertEqual(len(pred['prediction']), len(test_features))
        print('linear prediction:', pred['validation_error']['rmse_average'])

    def test_gp_quadratic_kernel(self):
        """Test Gaussian process predictions with the quadratic kernel."""
        train_features, train_targets, test_features, test_targets = get_data()

        # Test prediction routine with quadratic kernel.
        kdict = {'k1': {'type': 'quadratic', 'slope': 1., 'degree': 1.,
                        'scaling': 1.,
                        'bounds': ((1e-5, None),) *
                        (np.shape(train_features)[1] + 1),
                        'scaling_bounds': ((0., None),)}}
        gp = GaussianProcess(
            train_fp=train_features, train_target=train_targets,
            kernel_dict=kdict, regularization=1e-3,
            optimize_hyperparameters=True, scale_data=True)
        pred = gp.predict(test_fp=test_features,
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)
        self.assertEqual(len(pred['prediction']), len(test_features))
        print('quadratic prediction:',
              pred['validation_error']['rmse_average'])

    def test_gp_gaussian_kernel(self):
        """Test Gaussian process predictions with the gaussian kernel."""
        train_features, train_targets, test_features, test_targets = get_data()

        # Test prediction routine with gaussian kernel.
        kdict = {'k1': {'type': 'gaussian', 'width': 1., 'scaling': 1.}}
        gp = GaussianProcess(
            train_fp=train_features, train_target=train_targets,
            kernel_dict=kdict, regularization=1e-3,
            optimize_hyperparameters=True, scale_data=True)
        pred = gp.predict(test_fp=test_features,
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True,
                          uncertainty=True,
                          epsilon=0.1)
        mult_dim = pred['prediction']
        self.assertEqual(len(pred['prediction']), len(test_features))
        print('gaussian prediction (rmse):',
              pred['validation_error']['rmse_average'])
        print('gaussian prediction (ins):',
              pred['validation_error']['insensitive_average'])
        print('gaussian prediction (abs):',
              pred['validation_error']['absolute_average'])

        # Test prediction routine with single width parameter.
        kdict = {'k1': {'type': 'gaussian', 'width': 1., 'scaling': 1.,
                        'dimension': 'single',
                        'bounds': ((1e-5, None),),
                        'scaling_bounds': ((0., None),)}}
        gp = GaussianProcess(
            train_fp=train_features, train_target=train_targets,
            kernel_dict=kdict, regularization=1e-3,
            optimize_hyperparameters=True, scale_data=True)
        self.assertEqual(len(gp.kernel_dict['k1']['width']), 1)
        pred = gp.predict(test_fp=test_features,
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True,
                          uncertainty=True,
                          epsilon=0.1)
        self.assertEqual(len(pred['prediction']), len(test_features))
        self.assertNotEqual(np.sum(pred['prediction']), np.sum(mult_dim))
        print('gaussian single width (rmse):',
              pred['validation_error']['rmse_average'])

    def test_gp_laplacian_kernel(self):
        """Test Gaussian process predictions with the laplacian kernel."""
        train_features, train_targets, test_features, test_targets = get_data()

        # Test prediction routine with laplacian kernel.
        kdict = {'k1': {'type': 'laplacian', 'width': 1., 'scaling': 1.,
                        'bounds': ((1e-5, None),) *
                        np.shape(train_features)[1],
                        'scaling_bounds': ((0., None),)}}
        gp = GaussianProcess(
            train_fp=train_features, train_target=train_targets,
            kernel_dict=kdict, regularization=1e-3,
            optimize_hyperparameters=True, scale_data=True)
        pred = gp.predict(test_fp=test_features,
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)
        self.assertEqual(len(pred['prediction']), len(test_features))
        print('laplacian prediction:',
              pred['validation_error']['rmse_average'])

    def test_gp_addative_kernel(self):
        """Test Gaussian process predictions with the addative kernel."""
        train_features, train_targets, test_features, test_targets = get_data()

        # Test prediction with addative linear and gaussian kernel.
        kdict = {'k1': {'type': 'linear', 'features': [0, 1], 'scaling': 1.},
                 'k2': {'type': 'gaussian', 'features': [2, 3], 'width': 1.,
                        'scaling': 1.},
                 'c1': {'type': 'constant', 'const': 1.}}
        gp = GaussianProcess(
            train_fp=train_features, train_target=train_targets,
            kernel_dict=kdict, regularization=1e-3,
            optimize_hyperparameters=True, scale_data=True)
        pred = gp.predict(test_fp=test_features,
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)
        self.assertEqual(len(pred['prediction']), len(test_features))
        print('addition prediction:', pred['validation_error']['rmse_average'])

    def test_gp_multiplication_kernel(self):
        """Test Gaussian process predictions with the multiplication kernel."""
        train_features, train_targets, test_features, test_targets = get_data()

        # Test prediction with multiplication of linear & gaussian kernel.
        kdict = {'k1': {'type': 'linear', 'features': [0, 1], 'scaling': 1.},
                 'k2': {'type': 'gaussian', 'features': [2, 3], 'width': 1.,
                        'scaling': 1., 'operation': 'multiplication'},
                 'c1': {'type': 'constant', 'const': 1.}}
        gp = GaussianProcess(
            train_fp=train_features, train_target=train_targets,
            kernel_dict=kdict, regularization=1e-3,
            optimize_hyperparameters=True, scale_data=True)
        pred = gp.predict(test_fp=test_features,
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)
        self.assertEqual(len(pred['prediction']), len(test_features))
        print('multiplication prediction:',
              pred['validation_error']['rmse_average'])

    def test_gp_update(self):
        """Test Gaussian process predictions with the multiplication kernel."""
        train_features, train_targets, test_features, test_targets = get_data()

        kdict = {'k1': {'type': 'linear', 'scaling': 1.},
                 'k2': {'type': 'gaussian', 'width': 1., 'scaling': 1.,
                        'operation': 'multiplication'},
                 'c1': {'type': 'constant', 'const': 1.}}
        gp = GaussianProcess(
            train_fp=train_features, train_target=train_targets,
            kernel_dict=kdict, regularization=1e-3,
            optimize_hyperparameters=True, scale_data=True)

        # Test updating the last model.
        d, f = np.shape(train_features)
        train_features = np.concatenate((train_features, test_features))
        new_features = np.random.random_sample(
            (np.shape(train_features)[0], 5))
        train_features = np.concatenate((train_features, new_features), axis=1)
        self.assertNotEqual(np.shape(train_features), (d, f))
        train_targets = np.concatenate((train_targets, test_targets))
        new_features = np.random.random_sample((len(test_features), 5))
        test_features = np.concatenate((test_features, new_features), axis=1)
        kdict = {'k1': {'type': 'linear', 'scaling': 1.},
                 'k2': {'type': 'gaussian', 'width': 1., 'scaling': 1.,
                        'operation': 'multiplication'},
                 'c1': {'type': 'constant', 'const': 1.}}
        gp.update_gp(train_fp=train_features, train_target=train_targets,
                     kernel_dict=kdict)
        pred = gp.predict(test_fp=test_features,
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)
        self.assertEqual(len(pred['prediction']), len(test_features))
        print('Update prediction:',
              pred['validation_error']['rmse_average'])

    def test_gp_sensitivity(self):
        """Test Gaussian process predictions with sensitivity analysis."""
        train_features, train_targets, test_features, test_targets = get_data()

        # Start the sensitivity analysis.
        kdict = {'k1': {'type': 'gaussian', 'width': 30., 'scaling': 5.}}
        sen = SensitivityAnalysis(
            train_matrix=train_features[:, :5], train_targets=train_targets,
            test_matrix=test_features[:,
                                      :5], kernel_dict=kdict, init_reg=0.001,
            init_width=10.)

        sen.backward_selection(
            predict=True, test_targets=test_targets, selection=3)

    def test_general_gp(self):
        """Test the functions to build a general model."""
        train_features, train_targets, test_features, test_targets = get_data()

        ggp = GeneralGaussianProcess()

        ggp.train_gaussian_process(train_features, train_targets)
        pred = ggp.gaussian_process_predict(test_features)
        self.assertEqual(len(pred['prediction']), len(test_features))

        print('GeneralGP error: {0:.3f}'.format(
            get_error(pred['prediction'], test_targets)['rmse_average']))

        ggp = GeneralGaussianProcess(dimension='features')

        ggp.train_gaussian_process(train_features, train_targets)
        pred = ggp.gaussian_process_predict(test_features)
        self.assertEqual(len(pred['prediction']), len(test_features))

        print('GeneralGP error: {0:.3f}'.format(
            get_error(pred['prediction'], test_targets)['rmse_average']))


if __name__ == '__main__':
    unittest.main()
