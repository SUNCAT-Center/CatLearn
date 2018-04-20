"""Function performing GP sensitivity analysis."""
from __future__ import absolute_import
from __future__ import division

import numpy as np

from catlearn.regression import GaussianProcess
from catlearn.regression.gpfunctions.covariance import get_covariance


class SensitivityAnalysis(object):
    """Perform sensitivity analysis to estimate important features."""

    def __init__(self, train_matrix, train_targets, test_matrix,
                 kernel_dict, init_reg=0.001, init_width=10.):
        """Initialize the class.

        Parameters
        ----------
        train_matrix : array
            Training feature space data.
        train_targets : list
            A list of target values.
        test_matrix : array
            Testing feature space data.
        kernel_dict : dict
            Information for the kernel.
            sensitivities at each step.
        init_reg : float
            Specify the initial regularization strength.
        init_width : float
            Specify the initial kernel widths.
        """
        self.train_matrix = train_matrix
        self.train_targets = train_targets
        self.test_matrix = test_matrix
        self.kernel_dict = kernel_dict
        self.reg = init_reg
        self.width = init_width

    def backward_selection(self, predict=False, test_targets=None,
                           selection=None):
        """Feature selection with backward elimination.

        Parameters
        ----------
        predict : boolean
            Specify whether to make predictions on test data.
        test_targets : list
            A list of test targets to calculate errors, if known.
        selection : int, list
            Specify the number or range of features to consider.
        """
        self.predict = predict
        self.test_targets = test_targets

        original_index = list(range(np.shape(self.train_matrix)[1]))

        data = {}
        # Generate the initial model for all features.
        r = self._opt_step()
        data['original'], dlist = r, r['sorted_index']
        data['original']['abs_index'] = original_index

        # Set up the selection range.
        select1 = 1
        if selection is None:
            select2 = np.shape(self.train_matrix)[1]
        elif type(selection) is list:
            select1, select2 = selection[0], selection[1]
        else:
            select2 = selection

        pi = list(reversed(range(select1, select2)))
        dlist = data['original']['sorted_index']

        # Loop through features in reverse, eliminating one at a time.
        for i in pi:
            self.train_matrix = np.delete(self.train_matrix, dlist[i:], 1)
            self.test_matrix = np.delete(self.test_matrix, dlist[i:], 1)
            w = np.delete(self.kernel_dict['k1']['width'], dlist[i:])
            self.kernel_dict['k1']['width'] = w
            original_index = np.delete(original_index, dlist[i:])

            r = self._opt_step()
            data[str(i)], dlist = r, r['sorted_index']
            data[str(i)]['abs_index'] = original_index

        return data

    def _opt_step(self):
        """Helper function to get the feature sensitivity."""
        res = {'features': np.shape(self.train_matrix)[1]}
        # Generate the initial model for all features.
        model = self._get_opt_weights()
        if self.predict:
            p = self._make_predict(gp=model)
            res['prediction'] = p

        # Set optimal parameters.
        self.width = model.kernel_dict['k1']['width']
        res['widths'], res['reg'] = self.width, self.reg

        # Calculate the sensitivity of features.
        s = self._mean_sensitivity()

        sort_list = np.array(sorted(enumerate(s), key=lambda x: x[1],
                                    reverse=True))
        ind = np.array(sort_list)[:, :1]
        res['sorted_index'] = np.reshape(ind, (1, len(ind)))[0]
        res['sensitivity'] = s

        return res

    def _mean_sensitivity(self):
        """Feature sensitivity on the predicted mean."""
        d, f = np.shape(self.train_matrix)
        w = self.kernel_dict['k1']['width']

        # Calculate covariance.
        cvm = get_covariance(
            kernel_dict=self.kernel_dict, matrix1=self.train_matrix,
            regularization=self.reg, log_scale=False
        )
        ktb = get_covariance(
            kernel_dict=self.kernel_dict, matrix1=self.train_matrix,
            regularization=self.reg, log_scale=False
        )

        # Calculate weight estimates.
        cinv = np.linalg.inv(cvm)
        alpha = np.dot(cinv, self.train_targets)

        # Calculate sensitivities for all features.
        sen = []
        for j in range(f):
            d2 = 0.
            for q in self.test_matrix:
                d1 = 0.
                for p, ap, kt in zip(self.train_matrix, alpha, ktb):
                    d1 += np.sum(np.dot((np.dot(ap, np.subtract(p[j], q[j])) /
                                         w[j]), kt))
                d2 += 1 / d * (d1 ** 2)
            sen.append(d2)

        return np.asarray(sen)

    def _get_opt_weights(self):
        """Function to get optimized kernel weights."""
        # Train the GP.
        gp = GaussianProcess(
            train_fp=self.train_matrix, train_target=self.train_targets,
            kernel_dict=self.kernel_dict, regularization=self.reg,
            optimize_hyperparameters=True, scale_data=True
        )

        self.kernel_dict = gp.kernel_dict
        self.reg = gp.regularization

        return gp

    def _make_predict(self, gp):
        """Make prediction based on current feature representaion.

        Parameters
        ----------
        gp : object
            The optimized GP model.
        """
        ve = False
        if self.test_targets is not None:
            ve = True
        # Test data.
        pred = gp.predict(
            test_fp=self.test_matrix, test_target=self.test_targets,
            get_validation_error=ve, get_training_error=True
        )

        # print('{1} feature prediction ({0:.3f}):'.format(
        #    pred['validation_error']['rmse_average'],
        #    np.shape(self.test_matrix)[1]))

        return pred
