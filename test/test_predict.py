"""Script to test the prediction functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

from atoml.utilities import DescriptorDatabase
from atoml.regression import RidgeRegression, GaussianProcess

wkdir = os.getcwd()

train_size, test_size = 45, 5


def get_data():
    """Simple function to pull some training and test data."""
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
    train_features = feature_data[:train_size, :50]
    train_targets = target_data[:train_size]
    test_features = feature_data[test_size:, :50]
    test_targets = target_data[test_size:]

    return train_features, train_targets, test_features, test_targets


def rr_test(train_features, train_targets, test_features, test_targets):
    """Test ridge regression predictions."""
    # Test ridge regression predictions.
    rr = RidgeRegression(cv='loocv')
    reg = rr.find_optimal_regularization(X=train_features, Y=train_targets)
    coef = rr.RR(X=train_features, Y=train_targets, omega2=reg)[0]

    # Test the model.
    sumd = 0.
    for tf, tt in zip(test_features, test_targets):
        p = (np.dot(coef, tf))
        sumd += (p - tt) ** 2
    print('Ridge regression prediction:', (sumd / len(test_features)) ** 0.5)

    # Test ridge regression predictions.
    rr = RidgeRegression(cv='bootstrap')
    reg = rr.find_optimal_regularization(X=train_features, Y=train_targets)
    coef = rr.RR(X=train_features, Y=train_targets, omega2=reg)[0]

    # Test the model.
    sumd = 0.
    for tf, tt in zip(test_features, test_targets):
        p = (np.dot(coef, tf))
        sumd += (p - tt) ** 2
    print('Ridge regression prediction:', (sumd / len(test_features)) ** 0.5)


def gp_test(train_features, train_targets, test_features, test_targets):
    """Test Gaussian process predictions."""
    # Test prediction routine with linear kernel.
    kdict = {'k1': {'type': 'linear', 'scaling': 1.},
             'c1': {'type': 'constant', 'const': 1.}}
    gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True)
    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)
    assert len(pred['prediction']) == len(test_features)
    print('linear prediction:', pred['validation_error']['rmse_average'])

    # Test prediction routine with quadratic kernel.
    kdict = {'k1': {'type': 'quadratic', 'slope': 1., 'degree': 1.,
                    'scaling': 1.}}
    gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                         kernel_dict=kdict, regularization=-2,
                         optimize_hyperparameters=True)
    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)
    assert len(pred['prediction']) == len(test_features)
    print('quadratic prediction:', pred['validation_error']['rmse_average'])

    # Test prediction routine with gaussian kernel.
    kdict = {'k1': {'type': 'gaussian', 'width': 30., 'scaling': 10.}}
    gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True)
    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True,
                      uncertainty=True,
                      epsilon=0.1)
    no_scale = pred['prediction']
    assert len(pred['prediction']) == len(test_features)
    print('gaussian prediction (rmse):',
          pred['validation_error']['rmse_average'])
    print('gaussian prediction (ins):',
          pred['validation_error']['insensitive_average'])
    print('gaussian prediction (abs):',
          pred['validation_error']['absolute_average'])

    # Test prediction routine with default scaling.
    kdict = {'k1': {'type': 'gaussian', 'width': 30., 'scaling': 10.}}
    gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True, scale_data=True)
    assert len(gp.kernel_dict['k1']['width']) == np.shape(train_features)[1]
    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True,
                      uncertainty=True,
                      epsilon=0.1)
    assert len(pred['prediction']) == len(test_features)
    assert np.sum(pred['prediction']) != np.sum(no_scale)
    print('gaussian default scale (rmse):',
          pred['validation_error']['rmse_average'])

    # Test prediction routine with single width parameter.
    kdict = {'k1': {'type': 'gaussian', 'width': 30., 'scaling': 10.,
                    'dimension': 'single'}}
    gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True, scale_data=True)
    assert len(gp.kernel_dict['k1']['width']) == 1
    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True,
                      uncertainty=True,
                      epsilon=0.1)
    assert len(pred['prediction']) == len(test_features)
    assert np.sum(pred['prediction']) != np.sum(no_scale)
    print('gaussian single width (rmse):',
          pred['validation_error']['rmse_average'])

    # Test prediction routine with laplacian kernel.
    kdict = {'k1': {'type': 'laplacian', 'width': 1., 'scaling': 1.}}
    gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True)
    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)
    assert len(pred['prediction']) == len(test_features)
    print('laplacian prediction:', pred['validation_error']['rmse_average'])

    # Test prediction routine with addative linear and gaussian kernel.
    kdict = {'k1': {'type': 'linear', 'features': [0, 1], 'scaling': 1.},
             'k2': {'type': 'gaussian', 'features': [2, 3], 'width': 1.,
                    'scaling': 1.},
             'c1': {'type': 'constant', 'const': 0.}}
    gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True)
    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)
    assert len(pred['prediction']) == len(test_features)
    print('addition prediction:', pred['validation_error']['rmse_average'])

    # Test prediction routine with multiplication of linear & gaussian kernel.
    kdict = {'k1': {'type': 'linear', 'features': [0, 1], 'scaling': 1.},
             'k2': {'type': 'gaussian', 'features': [2, 3], 'width': 1.,
                    'scaling': 1., 'operation': 'multiplication'},
             'c1': {'type': 'constant', 'const': 0.}}
    gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True)
    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)
    assert len(pred['prediction']) == len(test_features)
    print('multiplication prediction:',
          pred['validation_error']['rmse_average'])

    # Test updating the last model.
    d, f = np.shape(train_features)
    train_features = np.concatenate((train_features, test_features))
    new_features = np.random.random_sample((np.shape(train_features)[0], 5))
    train_features = np.concatenate((train_features, new_features), axis=1)
    assert np.shape(train_features) != (d, f)
    train_targets = np.concatenate((train_targets, test_targets))
    new_features = np.random.random_sample((len(test_features), 5))
    test_features = np.concatenate((test_features, new_features), axis=1)
    kdict = {'k1': {'type': 'linear', 'features': [0, 1, 6, 9], 'scaling': 1.},
             'k2': {'type': 'gaussian', 'features': [2, 3, 4, 5], 'width': 1.,
                    'scaling': 1., 'operation': 'multiplication'},
             'c1': {'type': 'constant', 'const': 0.}}
    gp.update_gp(train_fp=train_features, train_target=train_targets,
                 kernel_dict=kdict)
    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)
    assert len(pred['prediction']) == len(test_features)
    print('Update prediction:',
          pred['validation_error']['rmse_average'])


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    train_features, train_targets, test_features, test_targets = get_data()
    rr_test(train_features, train_targets, test_features, test_targets)
    gp_test(train_features, train_targets, test_features, test_targets)

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
