""" Predictive KRR functions. """
import numpy as np
from math import exp
from collections import defaultdict


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

    def get_predictions(self, train_fp, test_fp, cinv, train_target,
                        test_target=None, uncertainty=False, basis=None,
                        get_validation_error=False, get_training_error=False):
        """ Returns a list of predictions for a test dataset.

            train_fp: list
                A list of the training fingerprint vectors.

            test_fp: list
                A list of the testing fingerprint vectors.

            cinv: matrix
                Covariance matrix for training dataset.

            train_target: list
                A list of the the training targets used to generate the
                predictions.

            test_target: list
                A list of the the test targets used to generate the
                prediction errors.

            uncertainty: boolean
                Return data on the predicted uncertainty if True. Default is
                False.

            basis: list
                set of basis functions to assess the reliability of the
                uncertainty predictions.

            get_validation_error: boolean
                Return the error associated with the prediction on the test set
                of data if True. Default is False.

            get_training_error: boolean
                Return the error associated with the prediction on the training
                set of data if True. Default is False.
        """
        if type(self.kwidth) is float:
            self.kwidth = np.zeros(len(train_fp[0]),) + self.kwidth

        data = defaultdict(list)
        ktb = np.asarray([[self.kernel(fp1=fp1, fp2=fp2) for fp1 in train_fp]
                          for fp2 in test_fp])

        # Build the list of predictions.
        data['prediction'] = self.do_prediction(ktb=ktb, cinv=cinv,
                                                target=train_target)

        # Calculate error associated with predictions on the training data.
        if get_training_error:
            kt_train = np.asarray([[self.kernel(fp1=fp1, fp2=fp2) for fp1 in
                                    train_fp] for fp2 in train_fp])
            data['train_prediction'] = self.do_prediction(ktb=kt_train,
                                                          cinv=cinv,
                                                          target=train_target)
            data['training_rmse'] = self.get_error(
                prediction=data['train_prediction'], actual=train_target)

        # Calculate uncertainty associated with prediction on test data.
        if uncertainty:
            data['uncertainty'] = self.get_uncertainty(train_fp=train_fp,
                                                       test_fp=test_fp,
                                                       ktb=ktb,
                                                       cinv=cinv,
                                                       target=train_target,
                                                       basis=basis,
                                                       pred=data['prediction'])

        # Calculate error associated with predictions on the test data.
        if get_validation_error:
            data['validation_rmse'] = self.get_error(
                prediction=data['prediction'], actual=test_target)

        return data

    def do_prediction(self, ktb, cinv, target):
        """ Function to do the actual prediction. """
        pred = []
        for kt in ktb:
            ktcinv = np.dot(kt, cinv)
            target_values = target
            train_mean = np.mean(target_values)
            target_values -= train_mean
            pred.append(np.dot(ktcinv, target_values) + train_mean)

        return pred

    def get_uncertainty(self, train_fp, test_fp, cinv, target, basis,
                        pred, ktb):
        """ Function to get the predicted uncertainty. Includes assessment with
            explicit basis functions defined above.
        """
        ud = defaultdict(list)
        if type(self.kwidth) is float:
            self.kwidth = np.zeros(len(train_fp[0]),) + self.kwidth
        ktest = np.asarray([[self.kernel(fp1=fp1, fp2=fp2)
                             for fp1 in test_fp] for fp2 in test_fp])

        # Form H and H* matrix, multiplying X by basis.
        mtrain = [i * basis for i in train_fp]
        mtest = [i * basis for i in test_fp]

        # Calculate R.
        r = mtest - np.dot(ktb, np.dot(cinv, mtrain))

        # Calculate beta.
        target_values = target
        train_mean = np.mean(target_values)
        target_values -= train_mean
        b = np.linalg.inv(np.dot(np.dot(cinv, mtrain), np.transpose(mtrain)))
        b = np.dot(target_values, np.dot(b, np.dot(cinv, mtrain)))

        fc = ktest - np.dot(np.dot(ktb, cinv), np.transpose(ktb))
        gca = np.dot(np.transpose(mtrain), np.dot(cinv, mtrain))
        ud['g_cov'] = fc + np.dot(r, np.dot(np.linalg.inv(gca),
                                            np.transpose(r)))

        # Do prediction.
        ud['gX'] = self.do_prediction(ktb=ktb, cinv=cinv,
                                      target=target) + np.dot(r, b)

        # Predict the uncertainty.
        ud['uncertainty'] = [(1 - np.dot(np.dot(kt, cinv),
                                         np.transpose(kt))) ** 0.5
                             for kt in ktb]

        return ud

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
            error['all'].append(e)
            sumd += e

        error['average'] = (sumd / len(prediction)) ** 0.5
        return error
