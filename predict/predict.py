""" Predictive functions for the machine learning genetic algorithm. """
import numpy as np
from math import exp
from collections import defaultdict


class FitnessPrediction(object):
    """ Kernel ridge regression functions for the machine learning genetic
        algorithm. This can be used to predict the fitness of a candidate
        within the GA run.

        ktype: string
            The kernel type, several have been pre-defined. Default is the
            Gaussian kernel.

        kwidth: float
            The kernel width, required for a number of the kernel types.
            Default is 0.5.

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
            return exp(-sum((fp1 - fp2) ** 2) /
                       (2 * len(fp1) * self.kwidth ** 2))

        # Laplacian kernel.
        elif self.ktype == 'laplacian':
            return exp(-sum(fp1 - fp2) / self.kwidth)

    def get_covariance(self, train_fp):
        """ Returns the covariance matrix between training dataset.

            train_fp: list
                A list of the training fingerprint vectors.
        """
        cov = np.asarray([[self.kernel(fp1=fp1, fp2=fp2)
                           for fp1 in train_fp] for fp2 in train_fp])
        if self.regularization is not None:
            cov = cov + self.regularization * np.identity(len(train_fp))
        covinv = np.linalg.inv(cov)

        return covinv

    def get_predictions(self, train_fp, test_fp, cinv, target, known=False,
                        test=None, key=None):
        """ Returns a list of predictions for a test dataset.

            train_fp: list
                A list of the training fingerprint vectors.

            test_fp: list
                A list of the testing fingerprint vectors.

            cinv: matrix
                Covariance matrix for training dataset.

            target: list
                A list of the targets used to generate the predictions.

            known: boolean
                If training raw_score is already known, the error of the
                predictions will be returned. Default is False.

            test: list
                A list of atoms objects which provided the test dataset. Only
                required to get error.

            key: string
                Property on which to base the predictions stored in the atoms
                object as atoms.info['key_value_pairs'][key]. Only required to
                get error.
        """
        data = defaultdict(list)
        # Do prediction.
        for tfp in test_fp:
            kt = [self.kernel(fp1=fp1, fp2=tfp) for fp1 in train_fp]
            ktcinv = np.dot(kt, cinv)
            target_values = target
            train_mean = np.mean(target_values)
            target_values -= train_mean
            predicted = np.dot(ktcinv, target_values) + train_mean

            # Build the list of predictions.
            data['prediction'].append(predicted)

        if known:
            # Calculate the error associated with the predictions.
            data['rmse'] = self.get_error(p=data['prediction'], k=test,
                                          key=key)

        return data

    def get_error(self, p, k, key):
        """ Returns the root mean squared error for predicted data relative to
            the actual fitnesses (Cost Function).

            p: list
                A list of predicted values.

            k: list
                A list of atoms objects that resulted in the test data. The
                target values are stored in k.info['key_value_pairs'][key].

            key: string
                property giving the target values.
        """
        assert len(p) == len(k)
        sumd = 0
        for i, j in zip(p, k):
            sumd += (i - j.info['key_value_pairs'][key]) ** 2

        return (sumd / len(p)) ** 0.5
