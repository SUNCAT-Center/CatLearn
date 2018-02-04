"""Script to test the prediction functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

from atoml.utilities import DescriptorDatabase
from atoml.preprocess.feature_preprocess import (standardize, normalize,
                                                 min_max, unit_length)
from atoml.preprocess.scale_target import (target_standardize,
                                           target_normalize, target_center)
from atoml.utilities.clustering import cluster_features

wkdir = os.getcwd()

train_size, test_size = 45, 5


def get_data():
    """Simple function to pull some training and test data."""
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
    test_targets = target_data[test_size:]

    return train_features, train_targets, test_features, test_targets


def scale_test(train_features, train_targets, test_features):
    """Test data scaling functions."""
    sfp = standardize(train_matrix=train_features, test_matrix=test_features)
    sfpg = standardize(train_matrix=train_features, test_matrix=test_features,
                       local=False)
    assert not np.allclose(sfp['train'], sfpg['train'])

    nfp = normalize(train_matrix=train_features, test_matrix=test_features)
    nfpg = normalize(train_matrix=train_features, test_matrix=test_features,
                     local=False)
    assert not np.allclose(nfp['train'], nfpg['train'])

    mmfp = min_max(train_matrix=train_features, test_matrix=test_features)
    mmfpg = min_max(train_matrix=train_features, test_matrix=test_features,
                    local=False)
    assert not np.allclose(mmfp['train'], mmfpg['train'])

    ulfp = unit_length(train_matrix=train_features, test_matrix=test_features)
    ulfpg = unit_length(train_matrix=train_features, test_matrix=test_features,
                        local=False)
    assert np.allclose(ulfp['train'], ulfpg['train'])

    ts = target_standardize(train_targets)
    assert not np.allclose(ts['target'], train_targets)

    ts = target_normalize(train_targets)
    assert not np.allclose(ts['target'], train_targets)

    ts = target_center(train_targets)
    assert not np.allclose(ts['target'], train_targets)


def cluster_test(train_features, train_targets, test_features, test_targets):
    """Test clustering function."""
    cf = cluster_features(train_matrix=train_features,
                          train_target=train_targets,
                          test_matrix=test_features,
                          test_target=test_targets,
                          k=2)
    assert len(cf['train_features']) == 2
    assert len(cf['train_target']) == 2
    assert len(cf['test_features']) == 2
    assert len(cf['test_target']) == 2


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    train_features, train_targets, test_features, test_targets = get_data()
    scale_test(train_features, train_targets, test_features)
    cluster_test(train_features, train_targets, test_features, test_targets)

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
