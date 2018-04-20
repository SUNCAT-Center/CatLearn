"""Some feature extraction routines."""
import numpy as np
from collections import defaultdict

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, SparsePCA

from .scaling import standardize
from .clean_data import clean_variance


def pls(components, train_matrix, target, test_matrix):
    """Projection of latent structure routine.

    Parameters
    ----------
    components : int
        The number of components to be returned.
    train_matrix : array
        The training features.
    test_matrix : array
        The test features.

    Returns
    -------
    new_train : array
        Extracted training features.
    new_test : array
        Extracted test features.
    """
    msg = 'The number of components must be a positive int greater than 0.'
    assert components > 0, msg

    pls = PLSRegression(n_components=components)
    model = pls.fit(X=train_matrix, Y=target)
    new_train = model.transform(train_matrix)
    new_test = model.transform(test_matrix)

    return new_train, new_test


def pca(components, train_matrix, test_matrix):
    """Principal component analysis routine.

    Parameters
    ----------
    components : int
        The number of components to be returned.
    train_matrix : array
        The training features.
    test_matrix : array
        The test features.

    Returns
    -------
    new_train : array
        Extracted training features.
    new_test : array
        Extracted test features.
    """
    msg = 'The number of components must be a positive int greater than 0.'
    assert components > 0, msg

    pca = PCA(n_components=components)
    model = pca.fit(X=train_matrix)
    new_train = model.transform(train_matrix)
    new_test = model.transform(test_matrix)

    return new_train, new_test


def spca(components, train_matrix, test_matrix):
    """Sparse principal component analysis routine.

    Parameters
    ----------
    components : int
        The number of components to be returned.
    train_matrix : array
        The training features.
    test_matrix : array
        The test features.

    Returns
    -------
    new_train : array
        Extracted training features.
    new_test : array
        Extracted test features.
    """
    msg = 'The number of components must be a positive int greater than 0.'
    assert components > 0, msg

    pca = SparsePCA(n_components=components)
    model = pca.fit(X=train_matrix)
    new_train = model.transform(train_matrix)
    new_test = model.transform(test_matrix)

    return new_train, new_test


def catlearn_pca(components, train_features, test_features=None, cleanup=False,
                 scale=False):
    """Principal component analysis varient that doesn't require scikit-learn.

    Parameters
    ----------
    components : int
        Number of principal components to transform the feature set by.
    test_fpv : array
        The feature matrix for the testing data.
    """
    msg = 'The number of components must be a positive int greater than 0.'
    assert components > 0, msg

    data = defaultdict(list)
    data['components'] = components
    if cleanup:
        c = clean_variance(train=train_features, test=test_features)
        test_features = c['test']
        train_features = c['train']
    if scale:
        std = standardize(train_matrix=train_features,
                          test_matrix=test_features)
        test_features = std['test']
        train_features = std['train']

    u, s, v = np.linalg.svd(np.transpose(train_features))

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(s[i]), u[:, i]) for i in range(len(s))]

    # Get the variance as percentage.
    data['variance'] = [(i / sum(s)) * 100 for i in sorted(s, reverse=True)]

    # Form the projection matrix.
    features = len(train_features[0])
    pm = eig_pairs[0][1].reshape(features, 1)
    if components > 1:
        for i in range(components - 1):
            pm = np.append(pm, eig_pairs[i][1].reshape(features, 1), axis=1)

    # Form feature matrix based on principal components.
    data['train_features'] = train_features.dot(pm)
    if test_features is not None:
        data['test_features'] = np.asarray(test_features).dot(pm)

    return data
