""" Functions to make predictions based on Gaussian Processes machine learning
    model.
"""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.optimize import minimize
from collections import defaultdict
from .model_selection import log_marginal_likelihood
from .output import write_predict
from .covariance import get_covariance
from .kernels import kdicts2list, list2kdict


class GaussianProcess(object):
    """ Kernel ridge regression functions for the machine learning. This can be
        used to predict the fitness of an atoms object.

        Parameters
        ----------
        kernel_dict    : dict of dicts
            Each dict in kernel_dict contains information on a kernel.
            The 'type' key is required to contain the name of kernel function:
            'linear', 'polynomial', 'gaussian' or 'laplacian'.
            The hyperparameters 'width', 'kfree'
        regularization : float
            The regularization strength (smoothing function) applied to the
            kernel matrix.
    """
    def __init__(self, kernel_dict, regularization=None):
        self.kernel_dict = kernel_dict
        self.regularization = regularization

    def prepare_kernels(self, N_D):
        kdict = self.kernel_dict
        for key in kdict:
            if 'features' in kdict[key]:
                N_D = len(kdict[key]['features'])
            # ktype = kdict[key]['type']
            if 'width' in kdict[key]:
                theta = kdict[key]['width']
                if type(theta) is float:
                    self.kernel_dict[key]['width'] = np.zeros(N_D,) + theta

            elif 'hyperparameters' in kdict[key]:
                theta = kdict[key]['hyperparameters']
                if type(theta) is float:
                    self.kernel_dict[key]['hyperparameters'] = (
                        np.zeros(N_D,) + theta)

            elif 'theta' in kdict[key]:
                theta = kdict[key]['theta']
                if type(theta) is float:
                    self.kernel_dict[key]['hyperparameters'] = (
                        np.zeros(N_D,) + theta)

    def get_predictions(self, train_fp, test_fp, train_target, cinv=None,
                        test_target=None, uncertainty=False, basis=None,
                        get_validation_error=False, get_training_error=False,
                        standardize_target=True, cost='squared', epsilon=None,
                        writeout=False, optimize_hyperparameters=False):
        """ Function to perform the prediction on some training and test data.

            Parameters
            ----------
            train_fp : list
                A list of the training fingerprint vectors.
            test_fp : list
                A list of the testing fingerprint vectors.
            train_target : list
                A list of the the training targets used to generate the
                predictions.
            cinv : matrix
                Covariance matrix for training dataset. Can be calculated on-
                the-fly or defined to utilize a model numerous times.
            test_target : list
                A list of the the test targets used to generate the
                prediction errors.
            uncertainty : boolean
                Return data on the predicted uncertainty if True. Default is
                False.
            basis : function
                Basis functions to assess the reliability of the uncertainty
                predictions. Must be a callable function that takes a list of
                descriptors and returns another list.
            get_validation_error : boolean
                Return the error associated with the prediction on the test set
                of data if True. Default is False.
            get_training_error : boolean
                Return the error associated with the prediction on the training
                set of data if True. Default is False.
            cost : str
                Define the way the cost function is calculated. Default is
                root mean squared error.
            epsilon : float
                Threshold for insensitive error calculation.
        """
        N_train, N_D = np.shape(train_fp)
        # Kernel dictionary should contain hyperparameters in lists:
        self.prepare_kernels(N_D)
        self.standardize_target = standardize_target

        error_train = train_target
        if standardize_target:
            self.standardize_data = target_standardize(train_target)
            train_target = self.standardize_data['target']

        data = defaultdict(list)
        data['input_kernels'] = self.kernel_dict
        data['input_regularization'] = self.regularization
        # Optimize hyperparameters.
        if optimize_hyperparameters:
            # Loop through kernels
            theta = kdicts2list(self.kernel_dict, N_D=N_D)
            if self.regularization is not None:
                theta = np.append(theta, self.regularization)

            # Define fixed arguments for log_marginal_likelihood
            args = (np.array(train_fp), train_target, self.kernel_dict)
            # Set bounds for hyperparameters
            bounds = ((1E-6, 1e3),)*(len(theta))
            # Optimize
            self.theta_opt = minimize(log_marginal_likelihood, theta,
                                      args=args,
                                      bounds=bounds)  # , jac=gradient_log_p)
            # Update kernel_dict and regularization
            self.kernel_dict = list2kdict(self.theta_opt['x'][:-1],
                                          self.kernel_dict)
            self.regularization = self.theta_opt['x'][-1]
            data['optimized_kernels'] = self.kernel_dict
            data['optimized_regularization'] = self.regularization

        # Get the Gram matrix on-the-fly if none is suppiled.
        if cinv is None:
            cvm = get_covariance(kernel_dict=self.kernel_dict,
                                 matrix1=train_fp,
                                 regularization=self.regularization)
            cinv = np.linalg.inv(cvm)

        # Calculate the covarience between the test and training datasets.
        ktb = get_covariance(kernel_dict=self.kernel_dict, matrix1=test_fp,
                             matrix2=train_fp, regularization=None)

        # Build the list of predictions.
        data['prediction'] = self.do_prediction(ktb=ktb, cinv=cinv,
                                                target=train_target)

        # Calculate error associated with predictions on the test data.
        if get_validation_error:
            data['validation_rmse'] = get_error(prediction=data['prediction'],
                                                target=test_target, cost=cost,
                                                epsilon=epsilon)

        # Calculate error associated with predictions on the training data.
        if get_training_error:
            # Calculate the covarience between the training dataset.
            kt_train = get_covariance(kernel_dict=self.kernel_dict,
                                      matrix1=train_fp, regularization=None)

            # Calculate predictions for the training data.
            data['train_prediction'] = self.do_prediction(ktb=kt_train,
                                                          cinv=cinv,
                                                          target=train_target)

            # Calculated the error for the prediction on the training data.
            data['training_rmse'] = get_error(
                prediction=data['train_prediction'], target=error_train,
                cost=cost, epsilon=epsilon)

        # Calculate uncertainty associated with prediction on test data.
        if uncertainty:
            data['uncertainty'] = [(1 - np.dot(np.dot(kt, cinv),
                                               np.transpose(kt))) ** 0.5 for
                                   kt in ktb]

        if basis is not None:
            data['basis_analysis'] = self.fixed_basis(train_fp=train_fp,
                                                      test_fp=test_fp,
                                                      ktb=ktb, cinv=cinv,
                                                      target=train_target,
                                                      test_target=test_target,
                                                      basis=basis, cost=cost,
                                                      epsilon=epsilon)

        if writeout:
            write_predict(function='get_predictions', data=data)

        return data

    def do_prediction(self, ktb, cinv, target):
        """ Function to make the prediction. """
        pred = []
        train_mean = np.mean(target)
        target_values = target - train_mean
        for kt in ktb:
            ktcinv = np.dot(kt, cinv)
            pred.append(np.dot(ktcinv, target_values) + train_mean)

        if self.standardize_target:
            pred = (np.asarray(pred) * self.standardize_data['std']) + \
             self.standardize_data['mean']

        return pred

    def fixed_basis(self, test_fp, train_fp, basis, ktb, cinv, target,
                    test_target, cost, epsilon):
        """ Function to apply fixed basis. Returns the predictions gX on the
            residual. """
        data = defaultdict(list)
        # Calculate the K(X*,X*) covarience matrix.
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

        # Form the covarience function based on the residual.
        covf = ktest - ktb.dot(cinv.dot(ktb.T))
        gca = train_matrix.T.dot(cinv.dot(train_matrix))
        data['g_cov'] = covf + r.dot(np.linalg.inv(gca).dot(r.T))

        # Do prediction accounting for basis.
        data['gX'] = self.do_prediction(ktb=ktb, cinv=cinv, target=target) + \
            beta.dot(r.T)

        # Calculated the error for the residual prediction on the test data.
        if test_target is not None:
            data['validation_rmse'] = get_error(prediction=data['gX'],
                                                target=test_target, cost=cost,
                                                epsilon=epsilon)

        return data


def target_standardize(target, writeout=False):
    """ Returns a list of standardized target values.

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


def get_error(prediction, target, cost='squared', epsilon=None):
    """ Returns the error for predicted data relative to target data. Discussed
        in: Rosasco et al, Neural Computation, (2004), 16, 1063-1076.

        Parameters
        ----------
        prediction : list
            A list of predicted values.
        target : list
            A list of target values.
    """
    msg = 'Something has gone wrong and there are '
    if len(prediction) < len(target):
        msg += 'more targets than predictions.'
    elif len(prediction) > len(target):
        msg += 'fewer targets than predictions.'
    assert len(prediction) == len(target), msg

    error = defaultdict(list)
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    if cost is 'squared':
        # Root mean squared error function.
        e = np.square(prediction - target)
        error['all'] = np.sqrt(e)
        error['average'] = np.sqrt(np.sum(e)/len(e))

    elif cost is 'absolute':
        # Absolute error function.
        e = np.abs(prediction - target)
        error['all'] = e
        error['average'] = np.sum(e)/len(e)

    elif cost is 'insensitive':
        # Epsilon-insensitive error function.
        e = np.abs(prediction - target) - epsilon
        np.place(e, e < 0, 0)
        error['all'] = e
        error['average'] = np.sum(e)/len(e)

    return error
