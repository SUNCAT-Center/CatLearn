"""Functions to make predictions with Gaussian Processes machine learning."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
import functools

from .model_selection import log_marginal_likelihood
from .output import write_predict
from .covariance import get_covariance
from .kernels import kdicts2list, list2kdict
from .uncertainty import get_uncertainty
from .cost_function import get_error


class GaussianProcess(object):
    """Gaussian processes functions for the machine learning."""

    def __init__(self, train_fp, train_target, kernel_dict,
                 regularization=None, optimize_hyperparameters=False):
        """Gaussian processes setup.

        Parameters
        ----------
        kernel_dict : dict of dicts
            Each dict in kernel_dict contains information on a kernel.
            The 'type' key is required to contain the name of kernel function:
            'linear', 'polynomial', 'gaussian' or 'laplacian'.
            The hyperparameters 'width', 'kfree'
        regularization : float
            The regularization strength (smoothing function) applied to the
            kernel matrix.
        """
        self.N_train, self.N_D = np.shape(train_fp)
        self.regularization = regularization
        self.prepare_kernels(kernel_dict)
        self.update_data(train_fp, train_target)
        if optimize_hyperparameters:
            self.optimize_hyperparameters()

    def update_data(self, train_fp, train_target, standardize_target=True):
        """Update the training matrix, targets and covariance matrix.

        Parameters
        ----------
        train_fp : list
            A list of training fingerprint vectors.
        train_target : list
            A list of training targets used to generate the predictions.
        cinv : matrix
            Covariance matrix for training dataset. Calculated on-the-fly or
            defined to utilize a model numerous times.
        """
        # Get the shape of the training dataset.
        self.N_train, self.N_D = np.shape(train_fp)
        # Store the training data in the GP
        self.train_fp = train_fp
        # Store standardized target values by default.
        self.standardize_target = standardize_target
        if self.standardize_target:
            self.standardize_data = target_standardize(train_target)
            self.train_target = self.standardize_data['target']
        else:
            self.train_target = train_target
        # Get the Gram matrix on-the-fly if none is suppiled.
        cvm = get_covariance(kernel_dict=self.kernel_dict,
                             matrix1=self.train_fp,
                             regularization=self.regularization)
        # Invert the covariance matrix.
        self.cinv = np.linalg.inv(cvm)

    def prepare_kernels(self, kernel_dict):
        """Format the kernel_dictionary.

        Parameters
        ----------
        kernel_dict : dict
            Dictionary containing all information for the kernels.
        """
        kdict = kernel_dict
        for key in kdict:
            if 'features' in kdict[key]:
                N_D = len(kdict[key]['features'])
            else:
                N_D = self.N_D
            if 'width' in kdict[key]:
                theta = kdict[key]['width']
                if type(theta) is float:
                    kernel_dict[key]['width'] = np.zeros(N_D,) + \
                                                     theta

            elif 'hyperparameters' in kdict[key]:
                theta = kdict[key]['hyperparameters']
                if type(theta) is float:
                    kernel_dict[key]['hyperparameters'] = (
                        np.zeros(N_D,) + theta)

            elif 'theta' in kdict[key]:
                theta = kdict[key]['theta']
                if type(theta) is float:
                    kernel_dict[key]['hyperparameters'] = (
                        np.zeros(N_D,) + theta)
        self.kernel_dict = kernel_dict

    def optimize_hyperparameters(self):
        """Optimize hyperparameters of the Gaussian Process.

        Performed with respect to the log marginal likelihood. Optimized
        hyperparameters are saved in the kernel dictionary. Finally, the
        covariance matrix is updated.
        """
        # Create a list of all hyperparameters.
        theta = kdicts2list(self.kernel_dict, N_D=self.N_D)
        if self.regularization is not None:
            theta = np.append(theta, self.regularization)

        # Define fixed arguments for log_marginal_likelihood
        args = (np.array(self.train_fp), self.train_target,
                self.kernel_dict)

        # Set bounds for hyperparameters
        bounds = ((1E-6, 1e3),) * (len(theta))

        # Optimize
        self.theta_opt = minimize(log_marginal_likelihood, theta,
                                  args=args,
                                  bounds=bounds)

        # Update kernel_dict and regularization with optimized values.
        if self.regularization is not None:
            self.kernel_dict = list2kdict(self.theta_opt['x'][:-1],
                                          self.kernel_dict)
            self.regularization = self.theta_opt['x'][-1]
        else:
            self.kernel_dict = list2kdict(self.theta_opt['x'],
                                          self.kernel_dict)
        # Make a new covariance matrix with the optimized hyperparameters.
        cvm = get_covariance(kernel_dict=self.kernel_dict,
                             matrix1=self.train_fp,
                             regularization=self.regularization)
        # Invert the covariance matrix.
        self.cinv = np.linalg.inv(cvm)

    def get_predictions(self, test_fp, test_target=None, uncertainty=False,
                        basis=None, get_validation_error=False,
                        get_training_error=False,
                        epsilon=None,
                        writeout=False):
        """Function to perform the prediction on some training and test data.

        Parameters
        ----------
        test_fp : list
            A list of testing fingerprint vectors.
        test_target : list
            A list of the the test targets used to generate the prediction
            errors.
        uncertainty : boolean
            Return data on the predicted uncertainty if True. Default is False.
        basis : function
            Basis functions to assess the reliability of the uncertainty
            predictions. Must be a callable function that takes a list of
            descriptors and returns another list.
        get_validation_error : boolean
            Return the error associated with the prediction on the test set of
            data if True. Default is False.
        get_training_error : boolean
            Return the error associated with the prediction on the training set
            of data if True. Default is False.
        epsilon : float
            Threshold for insensitive error calculation.
        """
        # Store input data.
        data = defaultdict(list)

        # Calculate the covariance between the test and training datasets.
        ktb = get_covariance(kernel_dict=self.kernel_dict, matrix1=test_fp,
                             matrix2=self.train_fp, regularization=None)

        # Build the list of predictions.
        data['prediction'] = self.do_prediction(ktb=ktb, cinv=self.cinv,
                                                target=self.train_target)

        # Calculate error associated with predictions on the test data.
        if get_validation_error:
            data['validation_error'] = get_error(prediction=data['prediction'],
                                                 target=test_target,
                                                 epsilon=epsilon)

        # Calculate error associated with predictions on the training data.
        if get_training_error:
            # Calculate the covariance between the training dataset.
            kt_train = get_covariance(kernel_dict=self.kernel_dict,
                                      matrix1=self.train_fp,
                                      regularization=None)

            # Calculate predictions for the training data.
            data['train_prediction'] = \
                self.do_prediction(ktb=kt_train, cinv=self.cinv,
                                   target=self.train_target)

            # Calculated the error for the prediction on the training data.
            if self.standardize_target:
                train_target = self.train_target * \
                    self.standardize_data['std'] + \
                    self.standardize_data['mean']
            else:
                train_target = self.train_target
            data['training_error'] = \
                get_error(prediction=data['train_prediction'],
                          target=train_target,
                          epsilon=epsilon)

        # Calculate uncertainty associated with prediction on test data.
        if uncertainty:
            data['uncertainty'] = get_uncertainty(kernel_dict=self.kernel_dict,
                                                  test_fp=test_fp,
                                                  reg=self.regularization,
                                                  ktb=ktb, cinv=self.cinv)

        if basis is not None:
            data['basis_analysis'] = self.fixed_basis(train_fp=self.train_fp,
                                                      test_fp=test_fp,
                                                      ktb=ktb, cinv=self.cinv,
                                                      target=self.train_target,
                                                      test_target=test_target,
                                                      basis=basis,
                                                      epsilon=epsilon)

        if writeout:
            write_predict(function='get_predictions', data=data)

        return data

    def do_prediction(self, ktb, cinv, target):
        """Function to make the prediction.

        Parameters
        ----------
        ktb : array
            Covariance matrix between test and training data.
        cinv : array
            Inverted Gram matrix, covariance between training data.
        target : list
            The target values for the training data.

        Returns
        -------
        pred : list
            The rescaled predictions for the test data.
        """
        train_mean = np.mean(target)
        target_values = target - train_mean
        alpha = np.dot(cinv, target_values)

        # Form list of the actual predictions.
        pred = functools.reduce(np.dot, (ktb, alpha)) + train_mean

        # Rescalse the predictions if targets were previously standardized.
        if self.standardize_target:
            pred = (np.asarray(pred) * self.standardize_data['std']) + \
             self.standardize_data['mean']

        return pred

    def fixed_basis(self, test_fp, train_fp, basis, ktb, cinv, target,
                    test_target, epsilon):
        """Function to apply fixed basis.

        Returns
        -------
            Predictions gX on the residual.
        """
        data = defaultdict(list)
        # Calculate the K(X*,X*) covariance matrix.
        ktest = get_covariance(kernel_dict=self.kernel_dict, matrix1=test_fp,
                               regularization=None)

        # Form H and H* matrix, multiplying X by basis.
        train_matrix = np.asarray([basis(i) for i in train_fp])
        test_matrix = np.asarray([basis(i) for i in test_fp])

        # Calculate R.
        r = test_matrix - ktb.dot(cinv.dot(train_matrix))

        # Calculate beta.
        b1 = np.linalg.inv(train_matrix.T.dot(cinv.dot(train_matrix)))
        b2 = np.asarray(target).dot(cinv.dot(train_matrix))
        beta = b1.dot(b2)

        # Form the covariance function based on the residual.
        covf = ktest - ktb.dot(cinv.dot(ktb.T))
        gca = train_matrix.T.dot(cinv.dot(train_matrix))
        data['g_cov'] = covf + r.dot(np.linalg.inv(gca).dot(r.T))

        # Do prediction accounting for basis.
        data['gX'] = self.do_prediction(ktb=ktb, cinv=cinv, target=target) + \
            beta.dot(r.T)

        # Calculated the error for the residual prediction on the test data.
        if test_target is not None:
            data['validation_error'] = get_error(prediction=data['gX'],
                                                 target=test_target,
                                                 epsilon=epsilon)

        return data


def target_standardize(target, writeout=False):
    """Return a list of standardized target values.

    Parameters
    ----------
    target : list
        A list of the target values.
    """
    target = np.asarray(target)

    data = defaultdict(list)
    data['mean'] = np.mean(target)
    data['std'] = np.std(target)
    data['target'] = (target - data['mean']) / data['std']

    if writeout:
        write_predict(function='target_standardize', data=data)

    return data
