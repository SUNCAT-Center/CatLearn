""" Predictive KRR functions. """
import numpy as np
from math import exp
from collections import defaultdict

from .data_setup import target_standardize
from .output import write_predict


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

        return cov

    def get_predictions(self, train_fp, test_fp, train_target, cinv=None,
                        test_target=None, uncertainty=False, basis=None,
                        get_validation_error=False, get_training_error=False,
                        standardize_target=True, writeout=True):
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

            basis: function
                Basis functions to assess the reliability of the uncertainty
                predictions. Must be a callable function that takes a list of
                descriptors and returns another list.

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
            error_train = train_target
            self.standardize_data = target_standardize(train_target)
            train_target = self.standardize_data['target']

        data = defaultdict(list)
        # Get the Gram matrix on-the-fly if none is suppiled.
        if cinv is None:
            cvm = self.get_covariance(train_fp)
            cinv = np.linalg.inv(cvm)

        # Calculate the covarience between the test and training datasets.
        ktb = np.asarray([[self.kernel(fp1=fp1, fp2=fp2) for fp1 in train_fp]
                          for fp2 in test_fp])

        # Build the list of predictions.
        data['prediction'] = self.do_prediction(ktb=ktb, cinv=cinv,
                                                target=train_target)

        # Calculate error associated with predictions on the test data.
        if get_validation_error:
            data['validation_rmse'] = self.get_error(
                prediction=data['prediction'], target=test_target)

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
                prediction=data['train_prediction'], target=error_train)

        # Calculate uncertainty associated with prediction on test data.
        if uncertainty:
            data['uncertainty'] = self.get_uncertainty(cinv=cinv, ktb=ktb)

        if basis is not None:
            data['basis_analysis'] = self.fixed_basis(train_fp=train_fp,
                                                      test_fp=test_fp,
                                                      ktb=ktb,
                                                      cinv=cinv,
                                                      target=train_target,
                                                      basis=basis)

        if writeout:
            write_predict(function='get_predictions', data=data)

        return data

    def do_prediction(self, ktb, cinv, target):
        """ Function to make the prediction. """
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

    def fixed_basis(self, test_fp, train_fp, basis, ktb, cinv, target):
        """ Function to apply fixed basis. """
        ud = defaultdict(list)
        # Calculate the K(X*,X*) covarience matrix.
        ktest = np.asarray([[self.kernel(fp1=fp1, fp2=fp2)
                             for fp1 in test_fp] for fp2 in test_fp])

        # Form H and H* matrix, multiplying X by basis.
        train_matrix = [basis(i) for i in train_fp]
        test_matrix = [basis(i) for i in test_fp]

        # Calculate R.
        r = test_matrix - np.dot(ktb, np.dot(cinv, train_matrix))

        # Calculate beta.
        bp = np.linalg.inv(np.dot(np.dot(cinv, train_matrix),
                                  np.transpose(train_matrix)))
        b = np.dot(target, np.dot(bp, np.dot(cinv, train_matrix)))

        covf = ktest - np.dot(np.dot(ktb, cinv), np.transpose(ktb))
        gca = np.dot(np.transpose(train_matrix), np.dot(cinv, train_matrix))
        ud['g_cov'] = covf + np.dot(np.dot(r, np.linalg.inv(gca)),
                                    np.transpose(r))

        # Do prediction accounting for basis.
        ud['gX'] = np.dot(b, np.transpose(r)) + self.do_prediction(ktb, cinv,
                                                                   target)

        return ud

    def get_error(self, prediction, target):
        """ Returns the root mean squared error for predicted data relative to
            the target data.

            prediction: list
                A list of predicted values.

            target: list
                A list of target values.
        """
        msg = 'Something has gone wrong and there are '
        if len(prediction) < len(target):
            msg += 'more targets than predictions.'
        elif len(prediction) > len(target):
            msg += 'fewer targets than predictions.'
        assert len(prediction) == len(target), msg
        error = defaultdict(list)
        sumd = 0
        for i, j in zip(prediction, target):
            e = (i - j) ** 2
            error['all'].append(e ** 0.5)
            sumd += e

        error['average'] = (sumd / len(prediction)) ** 0.5
        return error

    def log_marginal_likelyhood1(self, cov, cinv, y):
        """ Return the log marginal likelyhood.
        (Equation 5.8 in C. E. Rasmussen and C. K. I. Williams, 2006)
        """
        n = len(y)
        y = np.vstack(y)
        data_fit = -(np.dot(np.dot(np.transpose(y), cinv), y)/2.)[0][0]
        L = np.linalg.cholesky(cov)
        logdetcov = 0
        for l in range(len(L)):
            logdetcov += np.log(L[l, l])
        complexity = -logdetcov
        normalization = -n*np.log(2*np.pi)/2
        p = data_fit + complexity + normalization
        return p
