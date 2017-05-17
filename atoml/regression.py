""" Regression models to assess features using scikit-learn framework. """
import numpy as np
from collections import defaultdict

from sklearn.linear_model import Lasso

from .predict import get_error


def lasso(size, target, train_matrix, steps=None, alpha=None, min_alpha=1.e-8,
          max_alpha=1.e-1, max_iter=1e5, test_matrix=None, test_target=None):
    """ Use the scikit-learn implementation of lasso for feature selection. All
        features are ordered according to they corresponding coefficients.

        Parameters
        ----------
        size : int
            Number of features that should be returned.
        target : list
            List containg the target values.
        train_matrix : array
            An n x f array containg the training features.
        steps : int
            Number of steps to be taken in the penalty function.
        alpha : float
            Single penalty without looping over a range.
        min_alpha : float
            Starting penalty when searching over range. Default is 1.e-8.
        max_alpha : float
            Final penalty when searching over range. Default is 1.e-1.
        max_iter : float
            Maximum number of iterations taken minimizing the lasso function.
        test_matrix : array
            An n x f array containg the test features.
        test_target : list
            List containg the actual target values for testing.
    """
    select = defaultdict(list)

    if steps is not None:
        alpha_list = np.linspace(max_alpha, min_alpha, steps)
        for alpha in alpha_list:
            lasso = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=True,
                          normalize=True, selection='random')
            xy_lasso = lasso.fit(train_matrix, target)
            nz = len(xy_lasso.coef_) - (xy_lasso.coef_ == 0.).sum()
            if nz not in select['features']:
                if test_matrix is not None:
                    linear = xy_lasso.predict(test_matrix)
                    select['linear_error'].append(
                        get_error(prediction=linear,
                                  target=test_target)['average'])
                select['features'].append(nz)
            if nz >= size:
                break
        if 'linear_error' in select:
            mi = select['linear_error'].index(min(select['linear_error']))
            select['min_features'] = select['features'][mi]
    else:
        lasso = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=True,
                      normalize=True, selection='random')
        xy_lasso = lasso.fit(train_matrix, target)
        if test_matrix is not None:
            linear = xy_lasso.predict(test_matrix)
            select['linear_error'].append(
                get_error(prediction=linear, target=test_target)['average'])
    select['coefs'] = np.abs(xy_lasso.coef_)
    index = list(range(len(select['coefs'])))

    sort_list = [list(i) for i in zip(*sorted(zip(select['coefs'], index),
                                              key=lambda x: x[0],
                                              reverse=True))]
    # NOTE should check that have the desired number of none zero coefficients.
    select['order'] = sort_list[1]
    select['train_matrix'] = np.delete(train_matrix, sort_list[1][size:],
                                       axis=1)
    if test_matrix is not None:
        select['test_matrix'] = np.delete(test_matrix, sort_list[1][size:],
                                          axis=1)

    return select
