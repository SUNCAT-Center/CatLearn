""" Predictive KRR functions. """
import numpy as np
from math import exp
from collections import defaultdict

from .data_setup import target_standardize


class FitnessPrediction(object):
    """ Kernel ridge regression functions for the machine learning. This can be
        used to predict the fitness of an atoms object.

        ktype: string
            The kernel type, several have been pre-defined. Default is the
            Gaussian kernel.

        kwidth: float or list
            The kernel width, required for a number of the kernel types. If a
            float is supplied it is converted to a d-length array, containing a
            width for each descriptor. Default is 0.5.

        kfree: float
            Free parameter for the polynomial kernel, giving trading off for
            the influence of higher-order and lower-order terms in the
            polynomial. Default is homogeneous (c=0).

        kdegree: float
            Degree parameter for the polynomial kernel. Default is quadratic
            (d=2).

        regularization: float
            The regularization strength (smoothing function) applied to the
            kernel matrix.
    """

    def __init__(self, ktype='gaussian', kwidth=0.5, kfree=0., kdegree=2.,
                 regularization=None):
        self.ktype = ktype
        self.kwidth = kwidth
        self.kfree = kfree
        self.kdegree = kdegree
        self.regularization = regularization

    def kernel(self, fp1, fp2):
        """ Kernel functions taking two fingerprint vectors. """
        # Linear kernel.
        if self.ktype == 'linear':
            return sum(fp1 * fp2)

        # Polynomial kernel.
        elif self.ktype == 'polynomial':
            return(sum(fp1 * fp2) + self.kfree) ** self.kdegree

        # Gaussian kernel.
        elif self.ktype == 'gaussian':
            return exp(-sum(np.abs((fp1 - fp2) ** 2) / (2 * self.kwidth ** 2)))

        # Laplacian kernel.
        elif self.ktype == 'laplacian':
            return exp(-sum(np.abs((fp1 - fp2) / self.kwidth)))

    def get_covariance(self, train_fp):
        """ Returns the covariance matrix between training dataset.

            train_fp: list
                A list of the training fingerprint vectors.
        """
        if type(self.kwidth) is float:
            self.kwidth = np.zeros(len(train_fp[0]),) + self.kwidth

        cov = np.asarray([[self.kernel(fp1=fp1, fp2=fp2)
                           for fp1 in train_fp] for fp2 in train_fp])
        if self.regularization is not None:
            cov = cov + self.regularization * np.identity(len(train_fp))
        covinv = np.linalg.inv(cov)

        return covinv

    def get_predictions(self, train_fp, test_fp, train_target, cinv=None,
                        test_target=None, uncertainty=False,
                        get_validation_error=False, get_training_error=False,
                        standardize_target=True):
        """ Returns a list of predictions for a test dataset.

            train_fp: list
                A list of the training fingerprint vectors.

            test_fp: list
                A list of the testing fingerprint vectors.

            train_target: list
                A list of the the training targets used to generate the
                predictions.

            cinv: matrix
                Covariance matrix for training dataset. Can be calculated on-
                the-fly or defined to utilize a model numerous times.

            test_target: list
                A list of the the test targets used to generate the
                prediction errors.

            uncertainty: boolean
                Return data on the predicted uncertainty if True. Default is
                False.

            get_validation_error: boolean
                Return the error associated with the prediction on the test set
                of data if True. Default is False.

            get_training_error: boolean
                Return the error associated with the prediction on the training
                set of data if True. Default is False.
        """
        self.standardize_target = standardize_target
        if type(self.kwidth) is float:
            self.kwidth = np.zeros(len(train_fp[0]),) + self.kwidth
        if standardize_target:
            self.standardize_data = target_standardize(train_target)
            train_target = self.standardize_data['target']

        data = defaultdict(list)
        # Get the Gram matrix on-the-fly if none is suppiled.
        if cinv is None:
            cinv = self.get_covariance(train_fp)

        # Calculate the covarience between the test and training datasets.
        ktb = np.asarray([[self.kernel(fp1=fp1, fp2=fp2) for fp1 in train_fp]
                          for fp2 in test_fp])

        # Build the list of predictions.
        data['prediction'] = self.do_prediction(ktb=ktb, cinv=cinv,
                                                target=train_target)

        # Calculate error associated with predictions on the test data.
        if get_validation_error:
            data['validation_rmse'] = self.get_error(
                prediction=data['prediction'], actual=test_target)

        # Calculate error associated with predictions on the training data.
        if get_training_error:
            # Calculate the covarience between the training dataset.
            kt_train = np.asarray([[self.kernel(fp1=fp1, fp2=fp2) for fp1 in
                                    train_fp] for fp2 in train_fp])

            # Calculate predictions for the training data.
            data['train_prediction'] = self.do_prediction(ktb=kt_train,
                                                          cinv=cinv,
                                                          target=train_target)

            # Calculated the error for the prediction on the training data.
            data['training_rmse'] = self.get_error(
                prediction=data['train_prediction'], actual=train_target)

        # Calculate uncertainty associated with prediction on test data.
        if uncertainty:
            data['uncertainty'] = self.get_uncertainty(cinv=cinv, ktb=ktb)

        return data

    def do_prediction(self, ktb, cinv, target):
        """ Function to do the actual prediction. """
        pred = []
        for kt in ktb:
            ktcinv = np.dot(kt, cinv)
            target_values = target
            train_mean = np.mean(target_values)
            target_values -= train_mean
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

    def get_error(self, prediction, actual):
        """ Returns the root mean squared error for predicted data relative to
            the actual data.

            prediction: list
                A list of predicted values.

            actual: list
                A list of actual values.
        """
        assert len(prediction) == len(actual)
        error = defaultdict(list)
        sumd = 0
        for i, j in zip(prediction, actual):
            e = (i - j) ** 2
            error['all'].append(e ** 0.5)
            sumd += e

        error['average'] = (sumd / len(prediction)) ** 0.5
        return error
