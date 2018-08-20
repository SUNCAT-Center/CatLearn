"""Functions to make predictions with Gaussian Processes machine learning."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.optimize import minimize, basinhopping
from collections import defaultdict
import functools
import warnings
from .gpfunctions.log_marginal_likelihood import log_marginal_likelihood
from .gpfunctions.covariance import get_covariance
from .gpfunctions.kernel_setup import prepare_kernels, kdicts2list, list2kdict
from .gpfunctions.uncertainty import get_uncertainty
from .gpfunctions.default_scale import ScaleData
from .cost_function import get_error, _cost_function


class GaussianProcess(object):
    """Gaussian processes functions for the machine learning."""

    def __init__(self, train_fp, train_target, kernel_dict, gradients=None,
                 regularization=None, regularization_bounds=None,
                 optimize_hyperparameters=False, scale_optimizer=False,
                 scale_data=False):
        """Gaussian processes setup.

        Parameters
        ----------
        train_fp : list
            A list of training fingerprint vectors.
        train_target : list
            A list of training targets used to generate the predictions.
        kernel_dict : dict
            This dict can contain many other dictionarys, each one containing
            parameters for separate kernels.
            Each kernel dict contains information on a kernel such as:
            -   The 'type' key containing the name of kernel function.
            -   The hyperparameters, e.g. 'scaling', 'lengthscale', etc.
        gradients : list
            A list of gradients for all training data.
        regularization : float
            The regularization strength (smoothing function) applied to the
            covariance matrix.
        regularization_bounds : tuple
            Optional to change the bounds for the regularization.
        optimize_hyperparameters : boolean
            Optional flag to optimize the hyperparameters.
        scale_optimizer : boolean
            Flag to define if the hyperparameters are log scale for
            optimization.
        scale_data : boolean
            Scale the training and test features as well as target values.
            Default is False.
        """
        # Perform some sanity checks.
        msg = 'GP must be trained on more than one data point.'
        assert np.shape(train_fp)[0] > 1, msg
        msg = 'The number of data does not match the number of targets.'
        assert np.shape(train_fp)[0] == len(train_target), msg

        _, self.N_D = np.shape(train_fp)
        self.regularization = regularization
        self.gradients = gradients
        self.scale_optimizer = scale_optimizer
        self.scale_data = scale_data

        # Set flag for evaluating gradients.
        self.eval_gradients = False
        if self.gradients is not None:
            self.eval_gradients = True

        # Set bounds on regularization during hyperparameter optimization.
        if regularization_bounds is None:
            regularization_bounds = (1e-6, None)
            if self.eval_gradients:
                regularization_bounds = (1e-3, 1e3)

        self.kernel_dict, self.bounds = prepare_kernels(
            kernel_dict, regularization_bounds=regularization_bounds,
            eval_gradients=self.eval_gradients, N_D=self.N_D
        )

        self.update_data(train_fp, train_target, gradients=self.gradients,
                         scale_optimizer=scale_optimizer)

        if optimize_hyperparameters:
            self.optimize_hyperparameters()

    def predict(self, test_fp, test_target=None, uncertainty=False, basis=None,
                get_validation_error=False, get_training_error=False,
                epsilon=None):
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
        # Perform some sanity checks.
        if get_validation_error:
            msg = 'No test targets provided, can not return validation error.'
            assert test_target is not None, msg

        # Enforce np.array type for test data.
        test_fp = np.asarray(test_fp)
        if self.scale_data:
            test_fp = self.scaling.test(test_fp)
        if test_target is not None:
            test_target = np.asarray(test_target)

        # Store input data.
        data = defaultdict(list)

        # Calculate the covariance between the test and training datasets.
        ktb = get_covariance(kernel_dict=self.kernel_dict, matrix1=test_fp,
                             matrix2=self.train_fp, regularization=None,
                             log_scale=self.scale_optimizer,
                             eval_gradients=self.eval_gradients)

        # Build the list of predictions.
        data['prediction'] = self._make_prediction(ktb=ktb, cinv=self.cinv,
                                                   target=self.train_target)

        # Calculate error associated with predictions on the test data.
        if get_validation_error:
            data['validation_error'] = get_error(prediction=data['prediction'],
                                                 target=test_target,
                                                 epsilon=epsilon)

        # Calculate error associated with predictions on the training data.
        if get_training_error:
            # Calculate the covariance between the training dataset.
            kt_train = get_covariance(
                kernel_dict=self.kernel_dict, matrix1=self.train_fp,
                regularization=None, log_scale=self.scale_optimizer,
                eval_gradients=self.eval_gradients)

            # Calculate predictions for the training data.
            data['train_prediction'] = self._make_prediction(
                ktb=kt_train, cinv=self.cinv, target=self.train_target
            )

            # Calculated the error for the prediction on the training data.
            if self.scale_data:
                train_target = self.scaling.rescale_targets(self.train_target)
            else:
                train_target = self.train_target
            data['training_error'] = get_error(
                prediction=data['train_prediction'], target=train_target,
                epsilon=epsilon
            )

        # Calculate uncertainty associated with prediction on test data.
        if uncertainty:
            data['uncertainty'] = get_uncertainty(
                kernel_dict=self.kernel_dict, test_fp=test_fp,
                ktb=ktb, cinv=self.cinv,
                log_scale=self.scale_optimizer
            )

            data['uncertainty_with_reg'] = data['uncertainty'] + \
                self.regularization

            # Rescale uncertainty if needed.
            if self.scale_data:
                data['uncertainty'] *= self.scaling.target_data['std']
                data['uncertainty_with_reg'] *= self.scaling.target_data['std']

        if basis is not None:
            data['basis'] = self._fixed_basis(
                train=self.train_fp, test=test_fp, ktb=ktb, cinv=self.cinv,
                target=self.train_target, test_target=test_target, basis=basis,
                epsilon=epsilon
            )

        return data

    def predict_uncertainty(self, test_fp):
        """Return uncertainty only.

        Parameters
        ----------
        test_fp : list
            A list of testing fingerprint vectors.
        """
        # Calculate the covariance between the test and training datasets.
        ktb = get_covariance(kernel_dict=self.kernel_dict, matrix1=test_fp,
                             matrix2=self.train_fp, regularization=None,
                             log_scale=self.scale_optimizer,
                             eval_gradients=self.eval_gradients)
        # Store input data.
        data = defaultdict(list)

        data['uncertainty'] = get_uncertainty(
            kernel_dict=self.kernel_dict, test_fp=test_fp,
            ktb=ktb, cinv=self.cinv,
            log_scale=self.scale_optimizer)

        data['uncertainty_with_reg'] = data['uncertainty'] + \
            self.regularization

        # Rescale uncertainty if needed.
        if self.scale_data:
            data['uncertainty'] *= self.scaling.target_data['std']
            data['uncertainty_with_reg'] *= self.scaling.target_data['std']
        return data

    def update_data(self, train_fp, train_target=None, gradients=None,
                    scale_optimizer=False):
        """Update the training matrix, targets and covariance matrix.

        This function assumes that the descriptors in the feature set remain
        the same. That it is just the number of data ponts that is changing.
        For this reason the hyperparameters are not updated, so this update
        process should be fast.

        Parameters
        ----------
        train_fp : list
            A list of training fingerprint vectors.
        train_target : list
            A list of training targets used to generate the predictions.
        scale_optimizer : boolean
            Flag to define if the hyperparameters are log scale for
            optimization.
        """
        # Get the shape of the training dataset.
        d, f = np.shape(train_fp)

        # Perform some sanity checks.
        msg = 'GP must be trained on more than one data point.'
        assert d > 1, msg
        if self.N_D != f:
            msg = str(f) + '!=' + str(self.N_D)
            msg += '\n The number of features has changed. Train a new '
            msg += 'model instead of trying to update.'
            raise AssertionError(msg)

        # Store the training data in the GP, enforce np.array type.
        self.train_fp = np.asarray(train_fp)

        if train_target is not None:
            self.train_target = np.asarray(train_target)

        if self.scale_data:
            self.scaling = ScaleData(train_fp, train_target)
            self.train_fp, self.train_target = self.scaling.train()
            if gradients is not None:
                gradients = gradients / (self.scaling.target_data['std'] /
                                         self.scaling.feature_data['std'])
                gradients = np.ravel(gradients)

        if gradients is not None and train_target is not None:
            train_target_grad = np.append(self.train_target, gradients)
            self.train_target = np.reshape(train_target_grad,
                                           (np.shape(train_target_grad)[0], 1))

        # Get the Gram matrix on-the-fly if none is suppiled.
        cvm = get_covariance(
            kernel_dict=self.kernel_dict, matrix1=self.train_fp,
            regularization=self.regularization, log_scale=scale_optimizer,
            eval_gradients=self.eval_gradients)

        # Invert the covariance matrix.
        self.cinv = np.linalg.inv(cvm)
        if train_target is None:
            warnings.warn("GP mean not updated.")
            self.log_marginal_likelihood = np.nan
        else:
            self._update_lml()

    def optimize_hyperparameters(self, global_opt=False, algomin='L-BFGS-B',
                                 eval_jac=False, loss_function='lml'):
        """Optimize hyperparameters of the Gaussian Process.

        This function assumes that the descriptors in the feature set remain
        the same. Optimization is performed with respect to the log marginal
        likelihood. Optimized hyperparameters are saved in the kernel
        dictionary. Finally, the covariance matrix is updated.

        Parameters
        ----------
        global_opt : boolean
            Flag whether to do basin hopping optimization of hyperparameters.
            Default is False.
        algomin : str
            Define scipy minimizer method to call. Default is L-BFGS-B.
        """
        # Create a list of all hyperparameters.
        theta = kdicts2list(self.kernel_dict, N_D=self.N_D)
        theta = np.append(theta, self.regularization)

        if loss_function == 'lml':
            # Define fixed arguments for log_marginal_likelihood
            args = (np.array(self.train_fp), np.array(self.train_target),
                    self.kernel_dict, self.scale_optimizer,
                    self.eval_gradients, None, eval_jac)
            lf = log_marginal_likelihood
        elif loss_function == 'rmse' or loss_function == 'absolute':
            # Define fixed arguments for rmse loss function
            args = (np.array(self.train_fp), np.array(self.train_target),
                    self.kernel_dict, self.scale_optimizer, loss_function)
            lf = _cost_function
        else:
            raise NotImplementedError(str(loss_function))
        # Optimize
        if not global_opt:
            self.theta_opt = minimize(lf, theta,
                                      args=args,
                                      method=algomin,
                                      jac=eval_jac,
                                      bounds=self.bounds)
        else:
            minimizer_kwargs = {'method': algomin, 'args': args,
                                'bounds': self.bounds, 'jac': eval_jac,
                                'T': 10., 'interval': 30., 'niter': 30.}
            self.theta_opt = basinhopping(lf, theta,
                                          minimizer_kwargs=minimizer_kwargs)

        # Update kernel_dict and regularization with optimized values.
        self.kernel_dict = list2kdict(self.theta_opt['x'][:-1],
                                      self.kernel_dict)
        self.regularization = self.theta_opt['x'][-1]
        self.log_marginal_likelihood = -self.theta_opt['fun'][0]
        # Make a new covariance matrix with the optimized hyperparameters.
        cvm = get_covariance(kernel_dict=self.kernel_dict,
                             matrix1=self.train_fp,
                             regularization=self.regularization,
                             log_scale=self.scale_optimizer,
                             eval_gradients=self.eval_gradients)
        # Invert the covariance matrix.
        self.cinv = np.linalg.inv(cvm)

    def update_gp(self, train_fp=None, train_target=None, kernel_dict=None,
                  scale_optimizer=False, gradients=None,
                  regularization_bounds=(1e-6, None),
                  optimize_hyperparameters=False):
        """Potentially optimize the full Gaussian Process again.

        This alows for the definition of a new kernel as a result of changing
        descriptors in the feature space. Other parts of the model can also be
        changed. The hyperparameters will always be reoptimized.

        Parameters
        ----------
        train_fp : list
            A list of training fingerprint vectors.
        train_target : list
            A list of training targets used to generate the predictions.
        kernel_dict : dict
            This dict can contain many other dictionarys, each one containing
            parameters for separate kernels.
            Each kernel dict contains information on a kernel such as:
            -   The 'type' key containing the name of kernel function.
            -   The hyperparameters, e.g. 'scaling', 'lengthscale', etc.
        scale_optimizer : boolean
            Flag to define if the hyperparameters are log scale for
            optimization.
        regularization_bounds : tuple
            Optional to change the bounds for the regularization.
        """
        if train_fp is not None:
            msg = 'GP must be trained on more than one data point.'
            assert np.shape(train_fp)[0] > 1, msg
            _, self.N_D = np.shape(train_fp)
            self.train_fp = np.asarray(train_fp)

        # Assign flags for gradient evaluation.
        eval_gradients = False
        if gradients is not None:
            eval_gradients = True

        if kernel_dict is not None:
            self.kernel_dict, self.bounds = prepare_kernels(
                kernel_dict, regularization_bounds=regularization_bounds,
                eval_gradients=eval_gradients, N_D=self.N_D
            )
        if train_target is not None:
            msg = 'To update the data, both train_fp and train_target must be '
            msg += 'defined.'
            assert train_fp is not None, msg
            self.update_data(train_fp, train_target, gradients,
                             scale_optimizer)

        if optimize_hyperparameters:
            self.optimize_hyperparameters()
        else:
            self._update_lml()

    def _make_prediction(self, ktb, cinv, target):
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
            The predictions for the test data.
        """
        # Form list of the actual predictions.
        alpha = functools.reduce(np.dot, (cinv, target))

        pred = functools.reduce(np.dot, (ktb, alpha))

        if self.scale_data:
            pred = self.scaling.rescale_targets(pred)

        return pred

    def _fixed_basis(self, test, train, basis, ktb, cinv, target, test_target,
                     epsilon):
        """Function to apply fixed basis.

        Returns
        -------
            Predictions gX on the residual.
        """
        data = defaultdict(list)
        # Calculate the K(X*,X*) covariance matrix.
        ktest = get_covariance(
            kernel_dict=self.kernel_dict, matrix1=test, regularization=None,
            log_scale=self.scale_optimizer, eval_gradients=self.eval_gradients)

        # Form H and H* matrix, multiplying X by basis.
        train_matrix = np.asarray([basis(i) for i in train])
        test_matrix = np.asarray([basis(i) for i in test])

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
        data['gX'] = self._make_prediction(ktb=ktb, cinv=cinv, target=target) \
            + beta.dot(r.T)

        # Calculated the error for the residual prediction on the test data.
        if test_target is not None:
            data['validation_error'] = get_error(prediction=data['gX'],
                                                 target=test_target,
                                                 epsilon=epsilon)

        return data

    def _update_lml(self):
        # Create a list of all hyperparameters.
        theta = kdicts2list(self.kernel_dict, N_D=self.N_D)
        theta = np.append(theta, self.regularization)
        # Update log marginal likelihood.
        self.log_marginal_likelihood = -log_marginal_likelihood(
                theta=theta,
                train_matrix=np.array(self.train_fp),
                targets=np.array(self.train_target),
                kernel_dict=self.kernel_dict,
                scale_optimizer=self.scale_optimizer,
                eval_gradients=self.eval_gradients,
                cinv=self.cinv,
                eval_jac=False)
