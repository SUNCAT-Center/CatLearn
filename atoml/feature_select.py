""" Functions to select features for the fingerprint vectors. """
from __future__ import print_function

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import defaultdict
from math import log

from .fingerprint_setup import standardize
from .output import write_feature_select


def sure_independence_screening(target, train_fpv, size=None, writeout=False):
    """ Feature selection based on SIS discussed in Fan, J., Lv, J., J. R.
        Stat. Soc.: Series B, 2008, 70, 849.

        target: list
            The target values for the training data.

        train_fpv: array
            The feature matrix for the training data.

        size: int
            Number of features that should be left.

        std: boolean
            It is expected that the features and targets have been standardized
            prior to analysis. Automated if True.

        corr: str
            Correlation coefficient to use.
    """
    if size is not None:
        msg = 'Too few features avaliable, matrix cannot be reduced.'
        assert len(train_fpv[0]) >= size, msg

    select = defaultdict(list)

    omega = []
    for d in np.transpose(train_fpv):
        if all(d) == 0.:
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
    """ Correlation using kendall coefficients. """
    if size is not None:
        msg = 'Too few features avaliable, matrix cannot be reduced.'
        assert len(train_fpv[0]) >= size, msg

    select = defaultdict(list)

    omega = []
    if corr is 'kendall':
        for d in np.transpose(train_fpv):
            if all(d) == 0.:
                omega.append(0.)
            else:
                tau = kendalltau(x=d, y=target)[0]
                omega.append(tau - 0.25)
    elif corr is 'spearman':
        for d in np.transpose(train_fpv):
            if all(d) == 0.:
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


def iterative_screening(target, train_fpv, size=None, step=None, method='sis',
                        corr='kendall', writeout=True):
    """ Function to reduce the number of featues in an iterative manner using
        SIS or RRCS.

        step: int
            Step size by which to reduce the number of features. Default is
            n / log(n).

        method: str
            Specify the correlation method to be used, can be either sis or
            rrcs. Default is sis.

        corr: str
            Correlation to be used in rrcs, can be either kendall or spearman.
            Default is kendall.
    """
    # Assign default values for number of features to return and step size.
    if size is None:
        size = len(train_fpv)
    msg = 'Not enough features to perform iterative screening, reduce size.'
    assert len(train_fpv[0]) > size, msg
    if step is None:
        step = round(len(train_fpv) / log(len(train_fpv)))
    msg = 'Step size is too large, reduce it.'
    assert step < size

    select = defaultdict(list)
    ordering = list(range(len(train_fpv[0])))
    accepted = []

    # Initiate the feature reduction.
    if method is 'sis':
        screen = sure_independence_screening(target=target,
                                             train_fpv=train_fpv, size=step)
    elif method is 'rrcs':
        screen = robust_rank_correlation_screening(target=target,
                                                   train_fpv=train_fpv,
                                                   size=step, corr=corr)
    correlation = [screen['ordered_corr'][i] for i in screen['accepted']]
    accepted += [ordering[i] for i in screen['accepted']]
    ordering = [ordering[i] for i in screen['rejected']]
    reduced_fpv = np.delete(train_fpv, screen['rejected'], 1)
    train_fpv = np.delete(train_fpv, screen['accepted'], 1)

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
                                                 train_fpv=train_fpv,
                                                 size=step)
        elif method is 'rrcs':
            screen = robust_rank_correlation_screening(target=target,
                                                       train_fpv=train_fpv,
                                                       size=step, corr=corr)
        # Keep track of accepted and rejected features.
        correlation += [screen['ordered_corr'][i] for i in screen['accepted']]
        accepted += [ordering[i] for i in screen['accepted']]
        ordering = [ordering[i] for i in screen['rejected']]
        new_fpv = np.delete(train_fpv, screen['rejected'], 1)
        reduced_fpv = np.concatenate((reduced_fpv, new_fpv), axis=1)
        train_fpv = np.delete(train_fpv, screen['accepted'], 1)
    correlation += [screen['ordered_corr'][i] for i in screen['rejected']]

    select['correlation'] = correlation
    select['accepted'] = accepted
    select['rejected'] = ordering
    select['train_fpv'] = reduced_fpv

    if writeout:
        write_feature_select(function='iterative_screening', data=select)

    return select


def pca(components, train_fpv, test_fpv=None, writeout=True):
    """ Principal component analysis.

        components: int
            Number of principal components to transform the feature set by.

        test_fpv: array
            The feature matrix for the testing data.
    """
    data = defaultdict(list)
    data['components'] = components
    if test_fpv is not None:
        std = standardize(train=train_fpv, test=test_fpv, writeout=False)
        train_fpv = np.asarray(std['train'])
    else:
        train_fpv = standardize(train=train_fpv, writeout=False)['train']

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
        data['test_fpv'] = np.asarray(std['test']).dot(pm)

    if writeout:
        write_feature_select(function='pca', data=data)

    return data
