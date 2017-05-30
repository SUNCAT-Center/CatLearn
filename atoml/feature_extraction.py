import numpy as np
from collections import defaultdict

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, SparsePCA

from .feature_preprocess import standardize
from .utilities import clean_variance
from .output import write_feature_select


def pls(components, train_matrix, target, test_matrix):
    """ Projection of latent structure routine. """
    pls = PLSRegression(n_components=components)
    model = pls.fit(X=train_matrix, Y=target)
    new_train = model.transform(train_matrix)
    new_test = model.transform(test_matrix)

    return new_test, new_train


def pca(components, train_matrix, test_matrix):
    """ Principal component analysis routine. """
    pca = PCA(n_components=components)
    model = pca.fit(X=train_matrix)
    new_train = model.transform(train_matrix)
    new_test = model.transform(test_matrix)

    return new_test, new_train


def spca(components, train_matrix, test_matrix):
    """ Sparse principal component analysis routine. """
    pca = SparsePCA(n_components=components)
    model = pca.fit(X=train_matrix)
    new_train = model.transform(train_matrix)
    new_test = model.transform(test_matrix)

    return new_test, new_train


def home_pca(components, train_fpv, test_fpv=None, cleanup=False, scale=False,
             writeout=False):
    """ Principal component analysis varient that doesn't require scikit-learn.
        The results are not the same!

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
        c = clean_variance(train=train_fpv, test=test_fpv)
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
