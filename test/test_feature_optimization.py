"""Script to test feature space optimization functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import unittest

from atoml.utilities import DescriptorDatabase
from atoml.preprocess import importance_testing as it
from atoml.preprocess import feature_engineering as fe
from atoml.preprocess.feature_extraction import pls, pca, spca, atoml_pca
from atoml.preprocess.feature_elimination import FeatureScreening

wkdir = os.getcwd()

train_size, test_size = 30, 20


class TestFeatureOptimization(unittest.TestCase):
    """Test out the feature optimization."""

    def test_importance(self):
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

    def test_expand(self):
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
        self.assertTrue(np.shape(extend) == (d, f * 3))

        extend = fe.get_order_2(train_features)
        ext_n = fe.get_labels_order_2(names, div=False)
        self.assertTrue(np.shape(extend) == (d, f * (f + 1) / 2))
        self.assertTrue(len(ext_n) == np.shape(extend)[1])

        extend = fe.get_div_order_2(train_features)
        ext_n = fe.get_labels_order_2(names, div=True)
        self.assertTrue(np.shape(extend) == (d, f**2))
        self.assertTrue(len(ext_n) == np.shape(extend)[1])

        extend = fe.get_order_2ab(train_features, a=2, b=4)
        ext_n = fe.get_labels_order_2ab(names, a=2, b=4)
        self.assertTrue(np.shape(extend) == (d, f * (f + 1) / 2))
        self.assertTrue(len(ext_n) == np.shape(extend)[1])

        extend = fe.get_ablog(train_features, a=2, b=4)
        ext_n = fe.get_labels_ablog(names, a=2, b=4)
        self.assertTrue(np.shape(extend) == (d, f * (f + 1) / 2))
        self.assertTrue(len(ext_n) == np.shape(extend)[1])

        p = train_features[:3, :10]
        fe.generate_features(p, max_num=2, max_den=0, log=False,
                             sqrt=False, exclude=False, s=True)
        fe.generate_features(p, max_num=2, max_den=1, log=True,
                             sqrt=True, exclude=True, s=True)

        self.__class__.train_features = train_features
        self.__class__.train_targets = train_targets
        self.__class__.test_features = test_features

    def test_extract(self):
        """Test feature extraction."""
        nc = 3

        d = np.shape(self.train_features)[0]
        td = np.shape(self.test_features)[0]

        ext = pls(
            components=nc, train_matrix=self.train_features,
            target=self.train_targets, test_matrix=self.test_features)
        self.assertTrue(np.shape(ext[0]) == (d, nc) and np.shape(ext[1]) ==
                        (td, nc))

        ext = pca(
            components=nc, train_matrix=self.train_features,
            test_matrix=self.test_features)
        self.assertTrue(np.shape(ext[0]) == (d, nc) and np.shape(ext[1]) ==
                        (td, nc))

        # Sparse PCA is expensive, perform on reduced feature set.
        ext = spca(
            components=nc, train_matrix=self.train_features[:, :5],
            test_matrix=self.test_features[:, :5])
        self.assertTrue(np.shape(ext[0]) == (td, nc) and np.shape(ext[1]) ==
                        (td, nc))

        ext = atoml_pca(
            components=nc, train_features=self.train_features,
            test_features=self.test_features, cleanup=True, scale=True)
        self.assertTrue(np.shape(ext['train_features']) == (d, nc) and
                        np.shape(ext['test_features']) == (td, nc))

    def test_screening(self):
        """Test feature screening."""
        corr = ['pearson', 'spearman', 'kendall']
        d, f = np.shape(self.train_features)
        for c in corr:
            screen = FeatureScreening(correlation=c, iterative=False)
            feat = screen.eliminate_features(
                target=self.train_targets, train_features=self.train_features,
                test_features=self.test_features, size=d, step=None,
                order=None)
            self.assertTrue(np.shape(feat[0])[1] == d and np.shape(feat[1])[1]
                            == d)

            screen = FeatureScreening(correlation=c, iterative=True,
                                      regression='ridge')
            feat = screen.eliminate_features(
                target=self.train_targets, train_features=self.train_features,
                test_features=self.test_features, size=d, step=2, order=None)
            self.assertTrue(np.shape(feat[0])[1] == d and np.shape(feat[1])[1]
                            == d)

            screen = FeatureScreening(correlation=c, iterative=True,
                                      regression='lasso')
            feat = screen.eliminate_features(
                target=self.train_targets, train_features=self.train_features,
                test_features=self.test_features, size=d, step=2, order=None)
            self.assertTrue(np.shape(feat[0])[1] == d and np.shape(feat[1])[1]
                            == d)

            screen = FeatureScreening(correlation=c, iterative=True,
                                      regression='lasso', random_check=True)
            feat = screen.eliminate_features(
                target=self.train_targets, train_features=self.train_features,
                test_features=self.test_features, size=d, step=2, order=None)
            # Difficult to test this one as it is inherently random.


if __name__ == '__main__':
    unittest.main()
