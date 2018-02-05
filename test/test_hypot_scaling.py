"""Script to test the prediction functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

from atoml.utilities import DescriptorDatabase
from atoml.regression import GaussianProcess
from atoml.regression.gpfunctions import hyperparameter_scaling as hs

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
    train_features = feature_data[:train_size, :100]
    train_targets = target_data[:train_size]
    test_features = feature_data[test_size:, :100]
    test_targets = target_data[test_size:]

    return train_features, train_targets, test_features, test_targets


def gp_test(train_features, train_targets, test_features, test_targets):
    """Test Gaussian process predictions."""
    # Test prediction routine with gaussian kernel.
    kdict = {'k1': {'type': 'gaussian', 'width': 1., 'scaling': 1.}}
    gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True, scale_data=True)
    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True,
                      uncertainty=True)

    opt = gp.kernel_dict['k1']['width']
    orig = hs.rescale_hyperparameters(gp.scaling,
                                      gp.kernel_dict)['k1']['width']
    assert not np.allclose(opt, orig)
    scaled = hs.hyperparameters(gp.scaling, gp.kernel_dict)['k1']['width']
    assert np.allclose(opt, scaled)

    print('gaussian prediction (rmse):',
          pred['validation_error']['rmse_average'])


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    train_features, train_targets, test_features, test_targets = get_data()
    gp_test(train_features, train_targets, test_features, test_targets)

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
