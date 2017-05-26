""" Functions to select features for the fingerprint vectors. """
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import defaultdict
from math import log

from .regression import lasso
from .utilities import clean_variance
from .output import write_feature_select


def sure_independence_screening(target, train_fpv, size=None, writeout=False):
    """ Feature selection based on SIS discussed in Fan, J., Lv, J., J. R.
        Stat. Soc.: Series B, 2008, 70, 849.

        Parameters
        ----------
        target : list
            The target values for the training data.
        train_fpv : array
            The feature matrix for the training data.
        size : int
            Number of features that should be returned.
    """
    if size is not None:
        msg = 'Too few features avaliable, matrix cannot be reduced.'
        assert len(train_fpv[0]) > size, msg

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
                                      corr='kendall', writeout=False):
    """ Feature selection based on rank correlation coefficients. This can
        either be in the form of Kendall or Spearman's rank correlation.

        Parameters
        ----------
        target : list
            The target values for the training data.
        train_fpv : array
            The feature matrix for the training data.
        size : int
            Number of features that should be returned.
        corr : str
            Select correlation function, either kendall or spearman. Default is
            kendall.
    """
    if size is not None:
        msg = 'Too few features avaliable, matrix cannot be reduced.'
        assert len(train_fpv[0]) > size, msg

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
    rejected = list(range(f))
    accepted = []

    zd = []
    if cleanup:
        c = clean_variance(train=train_fpv, test=test_fpv)
        train_fpv = c['train']
        if 'index' in c:
            zd = c['index']
            zd.sort(reverse=True)
            for i in zd:
                del rejected[i]

    keep_train = train_fpv

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

    # Sort all new ordering of accepted and rejected features.
    sis_accept = [screen['accepted'][i] for i in lr['order'][:step]]
    sis_accept.sort(reverse=True)
    for i in sis_accept:
        accepted.append(rejected[i])
        del rejected[i]

    # Create new active feature matrix.
    train_fpv = np.delete(keep_train, accepted, axis=1)
    reduced_fpv = np.delete(keep_train, rejected, axis=1)

    # Iteratively reduce the remaining number of features by the step size.
    while len(reduced_fpv[0]) < size:
        if size - len(reduced_fpv[0]) < step:
            step = size - len(reduced_fpv[0])  # Calculate the remainder.
        # Calculate the residual for the remaining features.
        response = []
        for d in np.transpose(train_fpv):
            for a in np.transpose(reduced_fpv):
                r = (d - np.dot(a, np.dot(d, a))) / (np.linalg.norm(a) ** 2)
            response.append(r)
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

        # Sort all new ordering of accepted and rejected features.
        sis_accept = [screen['accepted'][i] for i in lr['order'][:step]]
        sis_accept.sort(reverse=True)
        for i in sis_accept:
            accepted.append(rejected[i])
            del rejected[i]

        # Create new active feature matrix.
        train_fpv = np.delete(keep_train, accepted, axis=1)
        reduced_fpv = np.delete(keep_train, rejected, axis=1)

    # Add on index of zero difference features removed at the start.
    rejected += zd

    if test_fpv is not None:
        test_fpv = np.delete(test_fpv, rejected, axis=1)
    if feature_names is not None:
        feature_names = list(np.delete(feature_names, rejected, axis=0))

    select['accepted'] = accepted
    select['rejected'] = rejected
    select['train_fpv'] = reduced_fpv
    select['test_fpv'] = test_fpv
    select['names'] = feature_names

    if writeout:
        write_feature_select(function='iterative_screening', data=select)

    return select
