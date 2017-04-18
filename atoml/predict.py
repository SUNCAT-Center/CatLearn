""" Functions to make predictions based on Gaussian Processes machine learning
    model.
"""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.spatial import distance
from collections import defaultdict

from .output import write_predict


class FitnessPrediction(object):
    """ Kernel ridge regression functions for the machine learning. This can be
        used to predict the fitness of an atoms object.

        Parameters
        ----------
        ktype : string
            The kernel type, several have been pre-defined. Default is the
            Gaussian kernel.
        kwidth : float or list
            The kernel width, required for a number of the kernel types. If a
            float is supplied it is converted to a d-length array, containing a
            width for each descriptor. Default is 0.5.
        kfree : float
            Free parameter for the polynomial kernel, giving trading off for
            the influence of higher-order and lower-order terms in the
            polynomial. Default is homogeneous (c=0).
        kdegree : float
            Degree parameter for the polynomial kernel. Default is quadratic
            (d=2).
        regularization : float
            The regularization strength (smoothing function) applied to the
            kernel matrix.
        combine_kernels : string
            Define how to combine kernels, can be addative or dot product.
        kernel_list : dict
            List of functions when combining kernels, each coupled with a List
            of features on which the given kernel should act. Example:
            {'gaussian': [1, 2, 5], 'linear': [0, 3, 4]}
        width_combine : dict
            List of feature widths, set up in same way as kernel_list.
    """

    def __init__(self, ktype='gaussian', kwidth=0.5, kfree=0., kdegree=2.,
                 regularization=None, combine_kernels=None, kernel_list=None,
                 width_combine=None):
        self.ktype = ktype
        self.kwidth = kwidth
        self.kfree = kfree
        self.kdegree = kdegree
        self.regularization = regularization
        self.combine_kernels = combine_kernels
        self.kernel_list = kernel_list
        self.width_combine = width_combine

    def kernel(self, m1, m2=None):
        """ Kernel functions taking n x d feature matrix.

            Parameters
            ----------
            m1 : array
                Feature matrix for training (or test) data.
            m2 : array
                Feature matrix for test data.

            Returns
            -------
            Kernelized representation of the feature space as array.
        """
        if m2 is None:
            # Gaussian kernel.
            if self.ktype is 'gaussian':
                k = distance.pdist(m1 / self.kwidth, metric='sqeuclidean')
                k = distance.squareform(np.exp(-.5 * k))
                np.fill_diagonal(k, 1)
                return k

            # Laplacian kernel.
            elif self.ktype is 'laplacian':
                k = distance.pdist(m1 / self.kwidth, metric='cityblock')
                k = distance.squareform(np.exp(-k))
                np.fill_diagonal(k, 1)
                return k

            # Otherwise set m2 equal to m1 as functions are the same.
            m2 = m1

        # Linear kernel.
        if self.ktype is 'linear':
            return np.dot(m1, np.transpose(m2))

        # Polynomial kernel.
        elif self.ktype is 'polynomial':
            return(np.dot(m1, np.transpose(m2)) + self.kfree) ** self.kdegree

        # Gaussian kernel.
        elif self.ktype is 'gaussian':
            k = distance.cdist(m1 / self.kwidth, m2 / self.kwidth,
                               metric='sqeuclidean')
            return np.exp(-.5 * k)

        # Laplacian kernel.
        elif self.ktype is 'laplacian':
            k = distance.cdist(m1 / self.kwidth, m2 / self.kwidth,
                               metric='cityblock')
            return np.exp(-k)

    def kernel_combine(self, m1, m2=None):
        """ Function to generate a covarience matric with a combination of
            kernel functions.

            Parameters
            ----------
            m1 : array
                Feature matrix for training (or test) data.
            m2 : array
                Feature matrix for test data.

            Returns
            -------
            Combined kernelized representation of the feature space as array.
        """
        msg = 'Must combine covarience from more than one kernel.'
        assert len(self.kernel_list) > 1, msg

        # Form addative covariance matrix.
        if self.combine_kernels is 'addative':
            if m2 is None:
                c = np.zeros((np.shape(m1)[0], np.shape(m1)[0]))
                f2 = m2
            else:
                c = np.zeros((np.shape(m1)[0], np.shape(m2)[0]))
            for k in self.kernel_list:
                self.kwidth = self.width_combine[k]
                f1 = m1[:, self.kernel_list[k]]
                if m2 is not None:
                    f2 = m2[:, self.kernel_list[k]]
                self.ktype = k
                c += self.kernel(m1=f1, m2=f2)
            return c

        # Form multliplication covariance matrix.
        if self.combine_kernels is 'multiplication':
            if m2 is None:
                c = np.ones((np.shape(m1)[0], np.shape(m1)[0]))
                f2 = m2
            else:
                c = np.ones((np.shape(m1)[0], np.shape(m2)[0]))
            for k in self.kernel_list:
                self.kwidth = self.width_combine[k]
                f1 = m1[:, self.kernel_list[k]]
                if m2 is not None:
                    f2 = m2[:, self.kernel_list[k]]
                self.ktype = k
                c *= self.kernel(m1=f1, m2=f2)
            return c

    def get_covariance(self, train_matrix):
        """ Returns the covariance matrix between training dataset.

            Parameters
            ----------
            train_matrix : list
                A list of the training fingerprint vectors.
        """
        if type(self.kwidth) is float:
            self.kwidth = np.zeros(len(train_matrix[0]),) + self.kwidth
        if self.width_combine is None and self.combine_kernels is not None:
            self.width_combine = {}
            for k in self.kernel_list:
                self.width_combine[k] = self.kwidth[self.kernel_list[k]]

        if self.combine_kernels is None:
            cov = self.kernel(m1=train_matrix, m2=None)
        else:
            cov = self.kernel_combine(m1=train_matrix, m2=None)

        if self.regularization is not None:
            cov = cov + self.regularization * np.identity(len(train_matrix))

        return cov

    def get_predictions(self, train_fp, test_fp, train_target, cinv=None,
                        test_target=None, uncertainty=False, basis=None,
                        get_validation_error=False, get_training_error=False,
                        standardize_target=True, cost='squared', epsilon=None,
                        writeout=False):
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
        self.standardize_target = standardize_target
        if type(self.kwidth) is float:
            self.kwidth = np.zeros(len(train_fp[0]),) + self.kwidth
        if self.width_combine is None and self.combine_kernels is not None:
            self.width_combine = {}
            for k in self.kernel_list:
                self.width_combine[k] = self.kwidth[self.kernel_list[k]]
        error_train = train_target
        if standardize_target:
            self.standardize_data = target_standardize(train_target)
            train_target = self.standardize_data['target']

        data = defaultdict(list)
        # Get the Gram matrix on-the-fly if none is suppiled.
        if cinv is None:
            cvm = self.get_covariance(train_fp)
            cinv = np.linalg.inv(cvm)

        # Calculate the covarience between the test and training datasets.
        if self.combine_kernels is None:
            ktb = self.kernel(m1=test_fp, m2=train_fp)
        else:
            ktb = self.kernel_combine(m1=test_fp, m2=train_fp)

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
            if self.combine_kernels is None:
                kt_train = self.kernel(m1=train_fp, m2=None)
            else:
                kt_train = self.kernel_combine(m1=train_fp, m2=None)

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
            data['uncertainty'] = self.get_uncertainty(cinv=cinv, ktb=ktb)

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
            p = np.dot(ktcinv, target_values) + train_mean
            if self.standardize_target:
                pred.append((p * self.standardize_data['std']) +
                            self.standardize_data['mean'])
            else:
                pred.append(p)

        return pred

    def get_uncertainty(self, cinv, ktb):
        """ Function to get the predicted uncertainty."""
        # Predict the uncertainty.
        return [(1 - np.dot(np.dot(kt, cinv), np.transpose(kt))) ** 0.5 for kt
                in ktb]

    def fixed_basis(self, test_fp, train_fp, basis, ktb, cinv, target,
                    test_target, cost, epsilon):
        """ Function to apply fixed basis. Returns the predictions gX on the
            residual. """
        data = defaultdict(list)
        # Calculate the K(X*,X*) covarience matrix.
        if self.combine_kernels is None:
            ktest = self.kernel(m1=test_fp, m2=None)
        else:
            ktest = self.kernel_combine(m1=test_fp, m2=None)

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
