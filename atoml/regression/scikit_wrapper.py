"""Regression models to assess features using scikit-learn framework."""
import numpy as np
from collections import defaultdict

from sklearn.linear_model import RidgeCV, Lasso, LassoCV, ElasticNetCV

from .cost_function import get_error


class RegressionFit(object):
    """Class to perform a fit to specified regression model."""

    def __init__(self, train_matrix, train_target, test_matrix=None,
                 test_target=None, method='ridge', predict=False):
        """Class setup.

        Parameters
        ----------
        train_matrix : array
            The feature matrix for the training data.
        train_target : list
            The target values for the training data.
        test_matrix : array
            The feature matrix for the test data.
        test_target : list
            The target values for the test data.
        method : str
            Define type of regression to fit model.
        predict : boolean
            Return the predicted error of the linear model. Default is False.
        """
        self.train_matrix = train_matrix
        self.train_target = train_target
        self.test_matrix = test_matrix
        self.test_target = test_target
        self.method = method
        self.predict = predict

    def feature_select(self, size=None, iterations=1e5, steps=None,
                       line_search=False, min_alpha=1.e-8, max_alpha=1.e-1,
                       eps=1e-3):
        """Find index of important featurs.

        Parameters
        ----------
        size : int
            Number best features to return.
        iterations : float
            Maximum number of iterations taken minimizing the regression
            function. Implemented in elastic net and lasso.
        steps : int
            Number of steps to be taken in the penalty function of LASSO.
        min_alpha : float
            Starting penalty when searching over range. Default is 1.e-8.
        max_alpha : float
            Final penalty when searching over range. Default is 1.e-1.
        """
        self.size = size
        self.iter = iterations
        if self.method is 'lasso':
            self.steps = steps
            self.line_search = line_search
            self.min_alpha = min_alpha
            self.max_alpha = max_alpha
            self.eps = eps

        select = defaultdict(list)

        select['coeff'], select['prediction'] = self._get_coefficients()
        index = list(range(len(select['coeff'])))
        sort_list = [list(i) for i in zip(*sorted(zip(np.abs(select['coeff']),
                                                      index),
                                                  key=lambda x: x[0],
                                                  reverse=True))]

        select['accepted'] = sort_list[1][:self.size]
        select['rejected'] = sort_list[1][self.size:]

        return select

    def _get_coefficients(self):
        """Function to get coefficients for features."""
        if self.method is 'ridge':
            coeff, pred = self._ridge()

        if self.method is 'elastic':
            coeff, pred = self._elast()

        if self.method is 'lasso':
            coeff, pred = self._lasso()

        return coeff, pred

    def _ridge(self):
        """Function to do ridge regression."""
        # Fit a linear ridge regression model.
        regr = RidgeCV(fit_intercept=True, normalize=True)
        model = regr.fit(X=self.train_matrix, y=self.train_target)
        coeff = regr.coef_

        # Make the linear prediction.
        pred = None
        if self.predict:
            data = model.predict(self.test_matrix)
            pred = get_error(prediction=data,
                             target=self.test_target)['average']

        return coeff, pred

    def _elast(self, tol=1e-4):
        """Function to do elastic net regression."""
        regr = ElasticNetCV(fit_intercept=True, normalize=True,
                            max_iter=self.iter, tol=tol)
        model = regr.fit(X=self.train_matrix, y=self.train_target)
        coeff = regr.coef_

        # Make the linear prediction.
        pred = None
        if self.predict:
            data = model.predict(self.test_matrix)
            pred = get_error(prediction=data,
                             target=self.test_target)['average']
        return coeff, pred

    def _lasso(self):
        """Order features according to their corresponding coefficients."""
        if self.line_search:
            pred = None
            try:
                alpha_list = np.geomspace(self.max_alpha, self.min_alpha,
                                          self.steps)
            except AttributeError:
                alpha_list = np.exp(np.linspace(np.log(self.max_alpha),
                                                np.log(self.min_alpha),
                                                self.steps))
            for alpha in alpha_list:
                regr = Lasso(alpha=alpha, max_iter=self.iter,
                             fit_intercept=True, normalize=True,
                             selection='random')
                model = regr.fit(self.train_matrix, self.train_target)
                nz = len(model.coef_) - (model.coef_ == 0.).sum()
                if nz >= self.size:
                    coeff = model.coef_
                    break
        else:
            regr = LassoCV(fit_intercept=True, normalize=True,
                           n_alphas=self.steps, max_iter=self.iter,
                           eps=self.eps, cv=None)
            model = regr.fit(X=self.train_matrix, y=self.train_target)
            coeff = model.coef_

            # Make the linear prediction.
            pred = None
            if self.predict:
                data = model.predict(self.test_matrix)
                pred = get_error(prediction=data,
                                 target=self.test_target)['average']

        return coeff, pred
