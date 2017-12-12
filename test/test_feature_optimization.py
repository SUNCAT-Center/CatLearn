"""Script to test feature space optimization functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

from atoml.utilities import DescriptorDatabase
from atoml.preprocess import feature_engineering as fe
from atoml.preprocess.feature_extraction import pls, pca, spca, atoml_pca
from atoml.preprocess.feature_elimination import FeatureScreening

wkdir = os.getcwd()

train_size, test_size = 45, 5


def test_extend():
    """Generate an extended feature space."""
    # Attach the database.
    dd = DescriptorDatabase(db_name='{}/fpv_store.sqlite'.format(wkdir),
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
    assert np.shape(extend) == (d, f*(f+1)/2)
    assert len(ext_n) == np.shape(extend)[1]

    extend = fe.get_div_order_2(train_features)
    ext_n = fe.get_labels_order_2(names, div=True)
    assert np.shape(extend) == (d, f**2)
    assert len(ext_n) == np.shape(extend)[1]

    extend = fe.get_order_2ab(train_features, a=2, b=4)
    ext_n = fe.get_labels_order_2ab(names, a=2, b=4)
    assert np.shape(extend) == (d, f*(f+1)/2)
    assert len(ext_n) == np.shape(extend)[1]

    extend = fe.get_ablog(train_features, a=2, b=4)
    ext_n = fe.get_labels_ablog(names, a=2, b=4)
    assert np.shape(extend) == (d, f*(f+1)/2)
    assert len(ext_n) == np.shape(extend)[1]

    return train_features, train_targets, test_features


def test_extract(train_features, train_targets, test_features):
    """Test feature extraction."""
    d, td = np.shape(train_features)[0], np.shape(test_features)[0]
    ext = pls(components=4, train_matrix=train_features, target=train_targets,
              test_matrix=test_features)
    assert np.shape(ext[0]) == (d, 4) and np.shape(ext[1]) == (td, 4)

    ext = pca(components=4, train_matrix=train_features,
              test_matrix=test_features)
    assert np.shape(ext[0]) == (d, 4) and np.shape(ext[1]) == (td, 4)

    ext = spca(components=4, train_matrix=train_features,
               test_matrix=test_features)
    assert np.shape(ext[0]) == (d, 4) and np.shape(ext[1]) == (td, 4)

    ext = atoml_pca(components=4, train_fpv=train_features,
                    test_fpv=test_features, cleanup=True, scale=True)
    assert np.shape(ext['train_fpv']) == (d, 4) and \
        np.shape(ext['test_fpv']) == (td, 4)


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
        assert np.shape(feat[0])[1] != d and np.shape(feat[1])[1] != d


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    train_features, train_targets, test_features = test_extend()
    test_extract(train_features, train_targets, test_features)
    test_screening(train_features, train_targets, test_features)

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
