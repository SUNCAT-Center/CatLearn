""" Functions to select features for the fingerprint vectors. """
from __future__ import print_function

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import defaultdict
from math import log

from .fingerprint_setup import standardize
from .output import write_feature_select
from .predict import get_error

sk_learn = False
try:
    from sklearn.linear_model import Lasso
except ImportError:
    sk_learn = True


def lasso(size, target, train_matrix, steps=None, alpha=1.e-5, min_alpha=1.e-8,
          max_alpha=1.e-1, max_iter=1e5, test_matrix=None, test_target=None,
          cleanup=False):
    """ Use the scikit-learn implementation of lasso for feature selection. """
    msg = "Must install scikit-learn to use this function:"
    msg += " http://scikit-learn.org/stable/"
    assert not sk_learn, msg

    if cleanup:
        c = clean_zero(train=train_matrix, test=test_matrix)
        test_matrix = c['test']
        train_matrix = c['train']

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


def sure_independence_screening(target, train_fpv, size=None, cleanup=False,
                                writeout=False):
    """ Feature selection based on SIS discussed in Fan, J., Lv, J., J. R.
        Stat. Soc.: Series B, 2008, 70, 849.

        Parameters
        ----------
        target : list
            The target values for the training data.
        train_fpv : array
            The feature matrix for the training data.
        size : int
            Number of features that should be left.
        cleanup : boolean
            Select whether to clean up the feature matrix. Default is False.
    """
    if size is not None:
        msg = 'Too few features avaliable, matrix cannot be reduced.'
        assert len(train_fpv[0]) > size, msg
    if cleanup:
        train_fpv = clean_zero(train_fpv)['train']

    select = defaultdict(list)

    omega = []
    for d in np.transpose(train_fpv):
        if np.allclose(d, d[0]):
            omega.append(0.)
        else:
            omega.append(pearsonr(x=d, y=target)[0])

    abso = [abs(i) for i in omega]
    order = list(range(np.shape(train_fpv)[1]))
    sort_list = [list(i) for i in zip(*sorted(zip(abso, order),
                                              key=lambda x: x[0],
                                              reverse=True))]

    select['sorted'] = sort_list[1]
    select['correlation'] = sort_list[0]
    select['ordered_corr'] = abso
    if size is not None:
        select['accepted'] = sort_list[1][:size]
        select['rejected'] = sort_list[1][size:]

    if writeout:
        write_feature_select(function='sure_independence_screening',
                             data=select)

    return select


def robust_rank_correlation_screening(target, train_fpv, size=None,
                                      corr='kendall', cleanup=False,
                                      writeout=False):
    """ Feature selection based on rank correlation coefficients. This can
        either be in the form of Kendall or Spearman's rank correlation.

        Parameters
        ----------
        target : list
            The target values for the training data.
        train_fpv : array
            The feature matrix for the training data.
        size : int
            Number of features that should be left.
        corr : str
            Select correlation function, either kendall or spearman. Default is
            kendall.
        cleanup : boolean
            Select whether to clean up the feature matrix. Default is False.
    """
    if size is not None:
        msg = 'Too few features avaliable, matrix cannot be reduced.'
        assert len(train_fpv[0]) > size, msg
    if cleanup:
        train_fpv = clean_zero(train_fpv)['train']

    select = defaultdict(list)

    omega = []
    if corr is 'kendall':
        for d in np.transpose(train_fpv):
            if np.allclose(d, d[0]):
                omega.append(0.)
            else:
                omega.append(kendalltau(x=d, y=target)[0])
    elif corr is 'spearman':
        for d in np.transpose(train_fpv):
            if np.allclose(d, d[0]):
                omega.append(0.)
            else:
                omega.append(spearmanr(a=d, b=target)[0])

    abso = [abs(i) for i in omega]
    order = list(range(np.shape(train_fpv)[1]))
    sort_list = [list(i) for i in zip(*sorted(zip(abso, order),
                                              key=lambda x: x[0],
                                              reverse=True))]

    select['sorted'] = sort_list[1]
    select['correlation'] = sort_list[0]
    select['ordered_corr'] = abso
    if size is not None:
        select['accepted'] = sort_list[1][:size]
        select['rejected'] = sort_list[1][size:]

    if writeout:
        write_feature_select(function='robust_rank_correlation_screening',
                             data=select)

    return select


def iterative_screening(target, train_fpv, test_fpv=None, size=None, step=None,
                        method='sis', corr='kendall', cleanup=False,
                        feature_names=None, writeout=False):
    """ Function to reduce the number of featues in an iterative manner using
        SIS or RRCS.

        Parameters
        ----------
        step : int
            Step size by which to reduce the number of features. Default is
            n / log(n).
        method : str
            Specify the correlation method to be used, can be either sis or
            rrcs. Default is sis.
        feature_names : list
            List of the feature names to be track useful features.
        cleanup : boolean
            Select whether to clean up the feature matrix. Default is True.
    """
    n = np.shape(train_fpv)[0]
    f = np.shape(train_fpv)[1]

    select = defaultdict(list)
    ordering = list(range(f))
    rejected = ordering
    keep_train = train_fpv

    zd = []
    if cleanup:
        c = clean_zero(train=train_fpv, test=test_fpv)
        train_fpv = c['train']
        test_fpv = c['test']
        if 'index' in c:
            zd = c['index']
            for i in zd:
                del rejected[i]

    # Assign default values for number of features to return and step size.
    if size is None:
        size = n
    msg = 'Not enough features to perform iterative screening, must be two '
    msg += 'times as many features compared to the size.'
    assert f > 2 * size, msg
    if step is None:
        step = round(n / log(n))
    msg = 'Step size is too large, reduce it.'
    assert step < size

    # Initiate the feature reduction.
    if method is 'sis':
        screen = sure_independence_screening(target=target,
                                             train_fpv=train_fpv, size=size)
    elif method is 'rrcs':
        screen = robust_rank_correlation_screening(target=target,
                                                   train_fpv=train_fpv,
                                                   size=size, corr=corr)
    screen_matrix = np.delete(train_fpv, screen['rejected'], axis=1)

    # Do LASSO down to step size.
    lr = lasso(size=step, target=target, train_matrix=screen_matrix,
               min_alpha=1.e-10, max_alpha=5.e-1, max_iter=1e5, steps=500)
    reduced_fpv = lr['train_matrix']
    accepted = [screen['accepted'][i] for i in lr['order'][:step]]
    train_fpv = np.delete(train_fpv, accepted, axis=1)
    for i in accepted:
        del rejected[i]

    # Iteratively reduce the remaining number of features by the step size.
    while len(reduced_fpv[0]) < size:
        if size - len(reduced_fpv[0]) < step:
            step = size - len(reduced_fpv[0])  # Calculate the remainder.
        # Calculate the residual for the remaining features.
        response = []
        for d in np.transpose(train_fpv):
            for a in np.transpose(reduced_fpv):
                d = (d - np.dot(a, np.dot(d, a))) / (np.linalg.norm(a) ** 2)
            response.append(d)
        response = np.transpose(response)

        # Do screening analysis on the residuals.
        if method is 'sis':
            screen = sure_independence_screening(target=target,
                                                 train_fpv=response,
                                                 size=size)
        elif method is 'rrcs':
            screen = robust_rank_correlation_screening(target=target,
                                                       train_fpv=response,
                                                       size=size, corr=corr)
        screen_matrix = np.delete(response, screen['rejected'], axis=1)

        # Do LASSO down to step size on remaining features.
        lr = lasso(size=step, target=target, train_matrix=screen_matrix,
                   min_alpha=1.e-10, max_alpha=5.e-1, max_iter=1e5, steps=500)
        reduced_fpv = np.concatenate((reduced_fpv, lr['train_matrix']), axis=1)
        sis_accept = [screen['accepted'][i] for i in lr['order'][:step]]
        new_accepted = [ordering[i] for i in sis_accept]
        train_fpv = np.delete(train_fpv, new_accepted, axis=1)
        for i in new_accepted:
            del rejected[i]
        accepted += new_accepted

    # Add on index of zero difference features removed at the start.
    rejected += zd

    if test_fpv is not None:
        select['test_fpv'] = np.delete(test_fpv, rejected, axis=1)
    if feature_names is not None:
        feature_names = list(np.delete(feature_names, rejected, axis=0))

    select['accepted'] = accepted
    select['rejected'] = rejected
    select['train_fpv'] = np.delete(keep_train, rejected, axis=1)
    select['names'] = feature_names

    if writeout:
        write_feature_select(function='iterative_screening', data=select)

    return select


def pca(components, train_fpv, test_fpv=None, cleanup=False, scale=False,
        writeout=False):
    """ Principal component analysis.

        Parameters
        ----------
        components : int
            Number of principal components to transform the feature set by.
        test_fpv : array
            The feature matrix for the testing data.
    """
    data = defaultdict(list)
    data['components'] = components
    if cleanup:
        c = clean_zero(train=train_fpv, test=test_fpv)
        test_fpv = c['test']
        train_fpv = c['train']
    if scale:
        std = standardize(train=train_fpv, test=test_fpv)
        test_fpv = std['test']
        train_fpv = std['train']

    u, s, v = np.linalg.svd(np.transpose(train_fpv))

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(s[i]), u[:, i]) for i in range(len(s))]

    # Get the varience as percentage.
    data['varience'] = [(i / sum(s))*100 for i in sorted(s, reverse=True)]

    # Form the projection matrix.
    features = len(train_fpv[0])
    pm = eig_pairs[0][1].reshape(features, 1)
    if components > 1:
        for i in range(components - 1):
            pm = np.append(pm, eig_pairs[i][1].reshape(features, 1), axis=1)

    # Form feature matrix based on principal components.
    data['train_fpv'] = train_fpv.dot(pm)
    if train_fpv is not None:
        data['test_fpv'] = np.asarray(test_fpv).dot(pm)

    if writeout:
        write_feature_select(function='pca', data=data)

    return data


def clean_zero(train, test=None):
    """ Function to remove features that contribute nothing to the model. """
    clean = defaultdict(list)
    m = train.T
    # Find features that provide no input for model.
    for i in list(range(len(m))):
        if np.allclose(m[i], m[i][0]):
            clean['index'].append(i)
    # Remove bad data from feature matrix.
    if 'index' in clean:
        train = np.delete(m, clean['index'], axis=0).T
        if test is not None:
            test = np.delete(test.T, clean['index'], axis=0).T
    clean['train'] = train
    clean['test'] = test

    return clean
