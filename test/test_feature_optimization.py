"""Script to test feature space optimization functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

from atoml.utilities import DescriptorDatabase
from atoml.preprocess import importance_testing as it
from atoml.preprocess import feature_engineering as fe
from atoml.preprocess.feature_extraction import pls, pca, spca, atoml_pca
from atoml.preprocess.feature_elimination import FeatureScreening
from atoml.preprocess.greedy_elimination import GreedyElimination
from atoml.regression import RidgeRegression

from common import get_data

wkdir = os.getcwd()

train_size, test_size = 30, 20


def test_importance():
    """Test feature importance helper functions."""
    # Attach the database.
    dd = DescriptorDatabase(db_name='{}/vec_store.sqlite'.format(wkdir),
                            table='FingerVector')

    # Pull the features and targets from the database.
    names = dd.get_column_names()
    feature_data = dd.query_db(names=names[1:-1])

    nf = it.feature_invariance(feature_data, 1)
    assert not np.allclose(nf, feature_data)
    nf = it.feature_randomize(feature_data, 1)
    assert not np.allclose(nf, feature_data)
    nf = it.feature_shuffle(feature_data, 1)
    assert not np.allclose(nf, feature_data)


def test_extend():
    """Generate an extended feature space."""
    # Attach the database.
    dd = DescriptorDatabase(db_name='{}/vec_store.sqlite'.format(wkdir),
                            table='FingerVector')

    # Pull the features and targets from the database.
    names = dd.get_column_names()
    features, targets = names[1:-1], names[-1:]
    feature_data = dd.query_db(names=features)
    target_data = np.reshape(dd.query_db(names=targets),
                             (np.shape(feature_data)[0], ))

    # Split the data into so test and training sets.
    train_features = feature_data[:train_size, :]
    train_targets = target_data[:train_size]
    test_features = feature_data[test_size:, :]
    d, f = np.shape(train_features)
    td, tf = np.shape(test_features)

    # Make some toy names.
    names = ['f{}'.format(i) for i in range(f)]

    # Perform feature engineering.
    extend = fe.single_transform(train_features)
    assert np.shape(extend) == (d, f * 3)

    extend = fe.get_order_2(train_features)
    ext_n = fe.get_labels_order_2(names, div=False)
    assert np.shape(extend) == (d, f * (f + 1) / 2)
    assert len(ext_n) == np.shape(extend)[1]

    extend = fe.get_div_order_2(train_features)
    ext_n = fe.get_labels_order_2(names, div=True)
    assert np.shape(extend) == (d, f**2)
    assert len(ext_n) == np.shape(extend)[1]

    extend = fe.get_order_2ab(train_features, a=2, b=4)
    ext_n = fe.get_labels_order_2ab(names, a=2, b=4)
    assert np.shape(extend) == (d, f * (f + 1) / 2)
    assert len(ext_n) == np.shape(extend)[1]

    extend = fe.get_ablog(train_features, a=2, b=4)
    ext_n = fe.get_labels_ablog(names, a=2, b=4)
    assert np.shape(extend) == (d, f * (f + 1) / 2)
    assert len(ext_n) == np.shape(extend)[1]

    p = train_features[:3, :10]
    fe.generate_features(p, max_num=2, max_den=0, log=False,
                         sqrt=False, exclude=False, s=True)
    fe.generate_features(p, max_num=2, max_den=1, log=True,
                         sqrt=True, exclude=True, s=True)

    return train_features, train_targets, test_features


def test_extract(train_features, train_targets, test_features):
    """Test feature extraction."""
    nc = 3

    d, td = np.shape(train_features)[0], np.shape(test_features)[0]

    ext = pls(components=nc, train_matrix=train_features, target=train_targets,
              test_matrix=test_features)
    assert np.shape(ext[0]) == (d, nc) and np.shape(ext[1]) == (td, nc)

    ext = pca(components=nc, train_matrix=train_features,
              test_matrix=test_features)
    assert np.shape(ext[0]) == (d, nc) and np.shape(ext[1]) == (td, nc)

    # Sparse PCA is expensive, perform on reduced feature set.
    ext = spca(components=nc, train_matrix=train_features[:, :5],
               test_matrix=test_features[:, :5])
    assert np.shape(ext[0]) == (td, nc) and np.shape(ext[1]) == (td, nc)

    ext = atoml_pca(components=nc, train_features=train_features,
                    test_features=test_features, cleanup=True, scale=True)
    assert np.shape(ext['train_features']) == (d, nc) and \
        np.shape(ext['test_features']) == (td, nc)


def test_screening(train_features, train_targets, test_features):
    """Test feature screening."""
    corr = ['pearson', 'spearman', 'kendall']
    d, f = np.shape(train_features)
    for c in corr:
        screen = FeatureScreening(correlation=c, iterative=False)
        feat = screen.eliminate_features(
            target=train_targets, train_features=train_features,
            test_features=test_features, size=d, step=None, order=None)
        assert np.shape(feat[0])[1] == d and np.shape(feat[1])[1] == d

        screen = FeatureScreening(correlation=c, iterative=True,
                                  regression='ridge')
        feat = screen.eliminate_features(
            target=train_targets, train_features=train_features,
            test_features=test_features, size=d, step=2, order=None)
        assert np.shape(feat[0])[1] == d and np.shape(feat[1])[1] == d

        screen = FeatureScreening(correlation=c, iterative=True,
                                  regression='lasso')
        feat = screen.eliminate_features(
            target=train_targets, train_features=train_features,
            test_features=test_features, size=d, step=2, order=None)
        assert np.shape(feat[0])[1] == d and np.shape(feat[1])[1] == d

        screen = FeatureScreening(correlation=c, iterative=True,
                                  regression='lasso', random_check=True)
        feat = screen.eliminate_features(
            target=train_targets, train_features=train_features,
            test_features=test_features, size=d, step=2, order=None)
        # Difficult to test this one as it is inherently random.


def prediction(train_features, train_targets, test_features, test_targets):
    """Ridge regression predictions."""
    # Test ridge regression predictions.
    rr = RidgeRegression(cv='loocv')
    reg = rr.find_optimal_regularization(X=train_features, Y=train_targets)
    coef = rr.RR(X=train_features, Y=train_targets, omega2=reg)[0]

    # Test the model.
    sumd = 0.
    for tf, tt in zip(test_features, test_targets):
        p = (np.dot(coef, tf))
        sumd += (p - tt) ** 2
    error = (sumd / len(test_features)) ** 0.5

    return error


def test_greedy(prediction, features, targets):
    """Test greedy feature selection."""
    ge = GreedyElimination()
    return ge.greedy_elimination(prediction, features, targets)


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    test_importance()
    train_features, train_targets, test_features = test_extend()
    test_extract(train_features, train_targets, test_features)
    test_screening(train_features, train_targets, test_features)

    train_features, train_targets, _, _ = get_data()
    result = test_greedy(prediction, train_features[:, :20], train_targets)

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
