"""Script to test the prediction functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

from atoml.utilities import DescriptorDatabase
from atoml.preprocess.feature_preprocess import (standardize, normalize,
                                                 min_max, unit_length)
from atoml.regression import RidgeRegression, GaussianProcess

wkdir = os.getcwd()

train_size, test_size = 45, 5


def scale_test():
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
    test_targets = target_data[test_size:]

    # Test ridge regression predictions.
    rr = RidgeRegression()
    reg = rr.find_optimal_regularization(X=train_features, Y=train_targets)
    coef = rr.RR(X=train_features, Y=train_targets, omega2=reg)[0]

    # Test the model.
    sumd = 0.
    for tf, tt in zip(test_features, test_targets):
        p = (np.dot(coef, tf))
        sumd += (p - tt) ** 2
    print('Ridge regression prediction:', (sumd / len(test_features)) ** 0.5)

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

    # Test prediction routine with linear kernel.
    kdict = {'k1': {'type': 'linear', 'scaling': 1.},
             'c1': {'type': 'constant', 'const': 0.}}
    gp = GaussianProcess(train_fp=sfp['train'], train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True)
    pred = gp.predict(test_fp=sfp['test'],
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)
    assert len(pred['prediction']) == len(sfp['test'])
    print('linear prediction:', pred['validation_error']['rmse_average'])

    # Test prediction routine with quadratic kernel.
    kdict = {'k1': {'type': 'quadratic', 'slope': 1., 'degree': 1.,
                    'scaling': 1.}}
    gp = GaussianProcess(train_fp=sfp['train'], train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True)
    pred = gp.predict(test_fp=sfp['test'],
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)
    assert len(pred['prediction']) == len(sfp['test'])
    print('quadratic prediction:', pred['validation_error']['rmse_average'])

    # Test prediction routine with gaussian kernel.
    kdict = {'k1': {'type': 'gaussian', 'width': 0.5, 'scaling': 1.}}
    gp = GaussianProcess(train_fp=sfp['train'], train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=False)
    pred = gp.predict(test_fp=sfp['test'],
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True,
                      uncertainty=True,
                      epsilon=0.1)
    assert len(pred['prediction']) == len(sfp['test'])
    print('gaussian prediction (rmse):',
          pred['validation_error']['rmse_average'])
    print('gaussian prediction (ins):',
          pred['validation_error']['insensitive_average'])
    print('gaussian prediction (abs):',
          pred['validation_error']['absolute_average'])

    # Test prediction routine with different scaling.
    scale = [sfp, sfpg, nfp, nfpg, mmfp, mmfpg, ulfp, ulfpg]
    name = ['standardize local', 'standardize global', 'normalize local',
            'normalize global', 'min max local', 'min max global',
            'unit length local', 'unit length global']
    for s, n in zip(scale, name):
        kdict = {'k1': {'type': 'gaussian', 'width': 1., 'scaling': 1.}}
        gp = GaussianProcess(train_fp=s['train'], train_target=train_targets,
                             kernel_dict=kdict, regularization=1e-3,
                             optimize_hyperparameters=True)
        pred = gp.predict(test_fp=s['test'],
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)
        assert len(pred['prediction']) == len(sfp['test'])
        print('gaussian prediction ({0}):'.format(n),
              pred['validation_error']['rmse_average'])

    # Test prediction routine with laplacian kernel.
    kdict = {'k1': {'type': 'laplacian', 'width': 0.5}}
    gp = GaussianProcess(train_fp=sfp['train'], train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True)
    pred = gp.predict(test_fp=sfp['test'],
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)
    assert len(pred['prediction']) == len(sfp['test'])
    print('laplacian prediction:', pred['validation_error']['rmse_average'])

    # Test prediction routine with addative linear and gaussian kernel.
    kdict = {'k1': {'type': 'linear', 'features': [0, 1]},
             'k2': {'type': 'gaussian', 'features': [2, 3], 'width': 0.5},
             'c1': {'type': 'constant', 'const': 0.}}
    gp = GaussianProcess(train_fp=sfp['train'], train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True)
    pred = gp.predict(test_fp=sfp['test'],
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)
    assert len(pred['prediction']) == len(sfp['test'])
    print('addition prediction:', pred['validation_error']['rmse_average'])

    # Test prediction routine with multiplication of linear & gaussian kernel.
    kdict = {'k1': {'type': 'linear', 'features': [0, 1]},
             'k2': {'type': 'gaussian', 'features': [2, 3], 'width': 0.5,
                    'operation': 'multiplication'},
             'c1': {'type': 'constant', 'const': 0.}}
    gp = GaussianProcess(train_fp=sfp['train'], train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True)
    pred = gp.predict(test_fp=sfp['test'],
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)
    assert len(pred['prediction']) == len(sfp['test'])
    print('multiplication prediction:',
          pred['validation_error']['rmse_average'])


if __name__ == '__main__':
    scale_test()
