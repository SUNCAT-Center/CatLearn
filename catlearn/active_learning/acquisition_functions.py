"""GP acquisition functions."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from scipy.stats import norm
from collections import defaultdict, Counter
from catlearn.utilities.clustering import cluster_features


def random_acquisition(y_best, predictions, uncertainty=None):
    """Return random numbers for control experiments.

    Parameters
    ----------
    y_best : float
        Condition
    predictions : list
        Predicted means.
    uncertainty : list
        Uncertainties associated with the predictions.
    """
    return np.random.rand(len(predictions))


def optimistic(y_best, predictions, uncertainty):
    """Find predictions that will optimistically lead to progress.

    Parameters
    ----------
    y_best : float
        Condition
    predictions : list
        Predicted means.
    uncertainty : list
        Uncertainties associated with the predictions.
    """
    metric = np.ravel(predictions) + np.ravel(uncertainty) - y_best
    return metric


def UCB(y_best, predictions, uncertainty, objective='max', kappa=1.5):
    """Upper-confidence bound acq. function.

    Parameters
    ----------
    y_best : float
        Condition
    predictions : list
        Predicted means.
    uncertainty : list
        Uncertainties associated with the predictions.
    kappa : float
        Constant that controls the explotation/exploration ratio in UCB.
    """
    if objective == 'max':
        return -(predictions - kappa * uncertainty)

    if objective == 'min':
        return -predictions + kappa * uncertainty


def EI(y_best, predictions, uncertainty, objective='max'):
    """Return expected improvement acq. function.

    Parameters
    ----------
    y_best : float
        Condition
    predictions : list
        Predicted means.
    uncertainty : list
        Uncertainties associated with the predictions.
    """
    if objective == 'max':
        z = (predictions - y_best) / (uncertainty)
        return (predictions - y_best) * norm.cdf(z) + \
            uncertainty * norm.pdf(
            z)

    if objective == 'min':
        z = (-predictions + y_best) / (uncertainty)
        return -((predictions - y_best) * norm.cdf(z) -
                 uncertainty * norm.pdf(z))


def PI(y_best, predictions, uncertainty, objective):
    """Probability of improvement acq. function.

    Parameters
    ----------
    y_best : float
        Condition
    predictions : list
        Predicted means.
    uncertainty : list
        Uncertainties associated with the predictions.
    """
    if objective == 'max':
        z = (predictions - y_best) / (uncertainty)
        return norm.cdf(z)

    if objective == 'min':
        z = -((predictions - y_best) / (uncertainty))
        return norm.cdf(z)


def proximity(y_best, predictions, uncertainty=None):
    """Return negative distances to y_best.

    Parameters
    ----------
    y_best : float
        Condition
    predictions : list
        Predicted means.
    uncertainty : list
        Uncertainties associated with the predictions.
    """
    metric = -np.abs(np.ravel(predictions) - y_best)
    return metric


def optimistic_proximity(y_best, predictions, uncertainty):
    """Return uncertainties minus distances to y_best.

    Parameters
    ----------
    y_best : float
        Condition
    predictions : list
        Predicted means.
    uncertainty : list
        Uncertainties associated with the predictions.
    """
    metric = np.ravel(uncertainty) - np.abs(np.ravel(predictions) - y_best)
    return metric


def probability_density(y_best, predictions, uncertainty):
    """Return probability densities at y_best.

    Parameters
    ----------
    y_best : float
        Condition
    predictions : list
        Predicted means.
    uncertainty : list
        Uncertainties associated with the predictions.
    """
    return norm.pdf(y_best, np.ravel(predictions), np.ravel(uncertainty))


def cluster(train_features, targets, test_features, predictions, k_means=3):
    """Penalize test points that are too clustered.

    Parameters
    ----------
    train_features : array
        Feature matrix for the training data.
    targets : list
        Training targets.
    test_features : array
        Feature matrix for the test data.
    predictions : list
        Predicted means.
    k_means : int
        Number of clusters.
    """
    fit = []

    cf = cluster_features(
        train_matrix=train_features, train_target=targets,
        k_means=k_means, test_matrix=test_features,
        test_target=predictions
    )

    train_count = Counter(cf['train_order'])

    for i, c in enumerate(cf['test_order']):
        fit.append(predictions[i] / train_count[c])

    return fit


def rank(targets, predictions, uncertainty, train_features=None,
         test_features=None, objective='max', k_means=3,
         kappa=1.5, metrics=['optimistic', 'UCB', 'EI', 'PI']):
    """Rank predictions based on acquisition function.

    Parameters
    ----------
    targets : list
        List of known target values.
    predictions : list
        List of predictions from the GP.
    uncertainty : list
        List of variance on the GP predictions.
    train_features : array
        Feature matrix for the training data.
    test_features : array
        Feature matrix for the test data.
    k_means : int
        Number of cluster to generate with clustering.
    kappa : float
        Constant that controls the explotation/exploration ratio in UCB.
    metrics : list
        list of strings.
        Accepted values are 'cdf', 'UCB', 'EI', 'PI', 'optimistic' and
        'pdf'.

    Returns
    -------
    res : dict
        A dictionary of lists containg the fitness of each test point for
        the different acquisition functions.

    """
    # Create dictionary for each acquisition function.
    res = {}

    # Select a fitness reference.
    if objective == 'max':
        y_best = max(targets)
    elif objective == 'min':
        y_best = min(targets)
    elif isinstance(objective, float):
        y_best = objective

    # Calculate fitness based on acquisition functions.
    if 'optimistic' in metrics:
        res['optimistic'] = optimistic(y_best, predictions, uncertainty)
    if 'UCB' in metrics:
        res['UCB'] = UCB(y_best, predictions, uncertainty, objective, kappa)
    if 'EI' in metrics:
        res['EI'] = EI(y_best, predictions, uncertainty, objective)
    if 'PI' in metrics:
        res['PI'] = PI(y_best, predictions, uncertainty, objective)
    if 'pdf' in metrics:
        res['pdf'] = probability_density(y_best, predictions, uncertainty)
    if 'cluster' in metrics:
        res['cluster'] = cluster(train_features, targets, test_features,
                                 predictions, k_means)

    return res


def classify(classifier, train_atoms, test_atoms, targets,
             predictions, uncertainty, train_features=None,
             test_features=None, objective='max',
             k_means=3, kappa=1.5,
             metrics=['optimistic', 'UCB', 'EI', 'PI']):
    """Classify ranked predictions based on acquisition function.

    Parameters
    ----------
    classifier : func
        User defined function to classify an atoms object.
    train_atoms : list
        List of atoms objects from training data upon which to base
        classification.
    test_atoms : list
        List of atoms objects from test data upon which to base
        classification.
    targets : list
        List of known target values.
    predictions : list
        List of predictions from the GP.
    uncertainty : list
        List of variance on the GP predictions.
    train_features : array
        Feature matrix for the training data.
    test_features : array
        Feature matrix for the test data.
    k_means : int
        Number of cluster to generate with clustering.
    kappa : float
        Constant that controls the explotation/exploration ratio in UCB.
    metrics : list
        list of strings.
        Accepted values are 'cdf', 'UCB', 'EI', 'PI', 'optimistic' and
        'pdf'.

    Returns
    -------
    res : dict
        A dictionary of lists containg the fitness of each test point for
        the different acquisition functions.
    """
    # Start by classifying the training data.
    best = defaultdict(list)
    for i, a in enumerate(train_atoms):
        c = classifier(a)
        best[c].append(targets[i])

    for i in best:
        # Select a fitness reference.
        if objective == 'max':
            best[i] = max(best[i])
            y_best = max(targets)
        elif objective == 'min':
            best[i] = min(best[i])
            y_best = min(targets)
        elif isinstance(objective, float):
            best[i] = objective
            y_best = objective

    test = defaultdict(dict)
    params = ['index', 'atoms', 'predictions', 'uncertainty',
              'train_features', 'test_features']
    # Calculate fitness based on acquisition functions.
    for i, a in enumerate(test_atoms):
        c = classifier(a)
        if c not in test:
            for key in params:
                test[c].update({key: []})
        test[c]['index'].append(i)
        test[c]['atoms'].append(a)
        # test[c]['targets'].append(targets[i])
        test[c]['predictions'].append(predictions[i])
        test[c]['uncertainty'].append(uncertainty[i])
        test[c]['train_features'].append(train_features[i])
        test[c]['test_features'].append(test_features[i])

    tmp_res = defaultdict(dict)
    for i in test:
        tmp_res[i]['index'] = test[i]['index']
        predictions = np.asarray(test[i]['predictions'])
        uncertainty = np.asarray(test[i]['uncertainty'])
        train_features = np.asarray(test[i]['train_features'])
        test_features = np.asarray(test[i]['test_features'])
        if 'optimistic' in metrics:
            tmp_res[i]['optimistic'] = optimistic(y_best, predictions,
                                                  uncertainty)
        if 'UCB' in metrics:
            tmp_res[i]['UCB'] = UCB(y_best, predictions, uncertainty,
                                    objective, kappa)
        if 'EI' in metrics:
            tmp_res[i]['EI'] = EI(y_best, predictions, uncertainty, objective)
        if 'PI' in metrics:
            tmp_res[i]['PI'] = PI(y_best, predictions, uncertainty, objective)
        if 'pdf' in metrics:
            tmp_res[i]['pdf'] = probability_density(y_best, predictions,
                                                    uncertainty)
        if 'cluster' in metrics:
            raise NotImplemented

    res = {}
    size = len(test_atoms)
    if 'optimistic' in metrics:
        res['optimistic'] = np.zeros(size)
    if 'UCB' in metrics:
        res['UCB'] = np.zeros(size)
    if 'EI' in metrics:
        res['EI'] = np.zeros(size)
    if 'PI' in metrics:
        res['PI'] = np.zeros(size)
    if 'pdf' in metrics:
        res['pdf'] = np.zeros(size)
    for i in tmp_res:
        for j, k in enumerate(tmp_res[i]['index']):
            if 'optimistic' in metrics:
                res['optimistic'][k] = tmp_res[i]['optimistic'][j]
            if 'UCB' in metrics:
                res['UCB'][k] = tmp_res[i]['UCB'][j]
            if 'EI' in metrics:
                res['EI'][k] = tmp_res[i]['EI'][j]
            if 'PI' in metrics:
                res['PI'][k] = tmp_res[i]['PI'][j]
            if 'pdf' in metrics:
                res['pdf'][k] = tmp_res[i]['pdf'][j]

    return res
