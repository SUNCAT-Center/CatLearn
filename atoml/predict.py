""" Predictive KRR functions. """
import numpy as np
from scipy.spatial import distance
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

    def kernel(self, m1, m2=None):
        """ Kernel functions taking n x d feature matrix. """
        if m2 is None:
            # Linear kernel.
            if self.ktype == 'linear':
                return np.dot(m1, np.transpose(m1))

            # Polynomial kernel.
            elif self.ktype == 'polynomial':
                return(np.dot(m1, np.transpose(m1)) + self.kfree) ** \
                 self.kdegree

            # Gaussian kernel.
            elif self.ktype == 'gaussian':
                k = distance.pdist(m1 / self.kwidth, metric='sqeuclidean')
                k = distance.squareform(np.exp(-.5 * k))
                np.fill_diagonal(k, 1)
                return k

            # Laplacian kernel.
            elif self.ktype == 'laplacian':
                k = distance.pdist(m1 / self.kwidth, metric='euclidean')
                k = distance.squareform(np.exp(-k))
                np.fill_diagonal(k, 1)
                return k

        # Linear kernel.
        elif self.ktype == 'linear':
            return np.dot(m1, np.transpose(m2))

        # Polynomial kernel.
        elif self.ktype == 'polynomial':
            return(np.dot(m1, np.transpose(m2)) + self.kfree) ** self.kdegree

        # Gaussian kernel.
        elif self.ktype == 'gaussian':
            k = distance.cdist(m1 / self.kwidth, m2 / self.kwidth,
                               metric='sqeuclidean')
            return np.exp(-.5 * k)

        # Laplacian kernel.
        elif self.ktype == 'laplacian':
            k = distance.cdist(m1 / self.kwidth, m2 / self.kwidth,
                               metric='euclidean')
            return np.exp(-k)

    def get_covariance(self, train_matrix):
        """ Returns the covariance matrix between training dataset.

            train_matrix: list
                A list of the training fingerprint vectors.
        """
        if type(self.kwidth) is float:
            self.kwidth = np.zeros(len(train_matrix[0]),) + self.kwidth

        cov = self.kernel(train_matrix)

        if self.regularization is not None:
            cov = cov + self.regularization * np.identity(len(train_matrix))

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
        ktb = self.kernel(test_fp, train_fp)

        # Build the list of predictions.
        data['prediction'] = self.do_prediction(ktb=ktb, cinv=cinv,
                                                target=train_target)

        # Calculate error associated with predictions on the test data.
        if get_validation_error:
            data['validation_rmse'] = get_error(prediction=data['prediction'],
                                                target=test_target)

        # Calculate error associated with predictions on the training data.
        if get_training_error:
            # Calculate the covarience between the training dataset.
            kt_train = self.kernel(train_fp, train_fp)

            # Calculate predictions for the training data.
            data['train_prediction'] = self.do_prediction(ktb=kt_train,
                                                          cinv=cinv,
                                                          target=train_target)

            # Calculated the error for the prediction on the training data.
            data['training_rmse'] = get_error(
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
                                                      test_target=test_target,
                                                      basis=basis)

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
                    test_target):
        """ Function to apply fixed basis. Returns the predictions gX on the
            residual. """
        data = defaultdict(list)
        # Calculate the K(X*,X*) covarience matrix.
        ktest = self.kernel(test_fp, test_fp)

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
                                                target=test_target)

        return data


def get_error(prediction, target):
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
