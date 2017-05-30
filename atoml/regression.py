"""Regression models to assess features using scikit-learn framework."""
import numpy as np
from collections import defaultdict

from sklearn.linear_model import (LinearRegression, RidgeCV, Lasso, LassoCV,
                                  ElasticNetCV)

from .predict import get_error


def ols(train_matrix, target, test_matrix=None, test_target=None):
    """Function to do Ordinary Least Squares regression."""
    regr = LinearRegression(fit_intercept=True, normalize=True)
    model = regr.fit(X=train_matrix, y=target)
    if test_matrix is not None:
        data = model.predict(test_matrix)
        print('LinearRegression',
              get_error(prediction=data, target=test_target)['average'])
    return data


def ridge(train_matrix, target, test_matrix=None, test_target=None):
    """Function to do Ordinary Least Squares regression."""
    regr = RidgeCV(fit_intercept=True, normalize=True)
    model = regr.fit(X=train_matrix, y=target)
    if test_matrix is not None:
        data = model.predict(test_matrix)
        print('RidgeCV',
              get_error(prediction=data, target=test_target)['average'])
    return data


def lass(train_matrix, target, test_matrix=None, test_target=None, steps=100,
         iter=1000, eps=1e-3, cv=None):
    """Function to do Ordinary Least Squares regression."""
    regr = LassoCV(fit_intercept=True, normalize=True, n_alphas=steps,
                   max_iter=iter, eps=eps, cv=cv)
    model = regr.fit(X=train_matrix, y=target)
    if test_matrix is not None:
        data = model.predict(test_matrix)
        print('LassoCV',
              get_error(prediction=data, target=test_target)['average'])

    order = list(range(1000))
    ind = []
    for i, j in zip(order, regr.coef_):
        if j != 0.:
            ind.append(i)
    print(len(regr.coef_[np.nonzero(regr.coef_)]), ind)

    return data


def elast(train_matrix, target, test_matrix=None, test_target=None, iter=1000,
          tol=1e-4):
    """Function to do Ordinary Least Squares regression."""
    regr = ElasticNetCV(fit_intercept=True, normalize=True, max_iter=iter,
                        tol=tol)
    model = regr.fit(X=train_matrix, y=target)
    if test_matrix is not None:
        data = model.predict(test_matrix)
        print('ElasticNetCV',
              get_error(prediction=data, target=test_target)['average'])
    return data


def lasso(size, target, train_matrix, steps=None, alpha=None, min_alpha=1.e-8,
          max_alpha=1.e-1, max_iter=1e5, test_matrix=None, test_target=None,
          spacing='linear'):
    """Order features according to their corresponding coefficients.

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
        if spacing is 'linear':
            alpha_list = np.linspace(max_alpha, min_alpha, steps)
        else:
            alpha_list = np.geomspace(max_alpha, min_alpha, steps)
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
