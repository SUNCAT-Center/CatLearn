"""Test io functions."""
import os
import numpy as np

from atoml.regression import GaussianProcess
from atoml.regression.gpfunctions import io
from common import get_data

wkdir = os.getcwd()


def train_model(train_features, train_targets):
    """Function to train a Gaussian process."""
    kdict = {
        'k1': {'type': 'gaussian', 'width': 1., 'scaling': 1.},
        'k2': {'type': 'linear', 'scaling': 1.},
        # 'k3': {'type': 'constant', 'const': 1.},
        'k4': {'type': 'quadratic', 'slope': 1., 'degree': 1., 'scaling': 1.},
    }
    gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True, scale_data=False)

    io.write(filename='test-model', model=gp, ext='pkl')
    io.write(filename='test-model', model=gp, ext='hdf5')

    return gp


def test_model(gp, test_features, test_targets):
    """Function to test Gaussian process."""
    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)

    return pred


def test_load(original, test_features, test_targets):
    """Function to teast loading a pre-generated model."""
    gp = io.read(filename='test-model', ext='pkl')

    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)

    assert np.allclose(pred['validation_error']['rmse_all'],
                       original['validation_error']['rmse_all'])

    # gp = io.read(filename='test-model', ext='hdf5')

    # pred = gp.predict(test_fp=test_features,
    #                  test_target=test_targets,
    #                  get_validation_error=True,
    #                  get_training_error=True)

    # print(pred['validation_error']['rmse_all'])
    # print(original['validation_error']['rmse_all'])

    # assert np.allclose(pred['validation_error']['rmse_all'],
    #                   original['validation_error']['rmse_all'])

    os.remove('{}/test-model.pkl'.format(wkdir))
    # os.remove('{}/test-model.hdf5'.format(wkdir))


def test_raw(train_features, train_targets, regularization, kernel_dict):
    """Function to test raw data save."""
    io.write_train_data('train_data', train_features, train_targets,
                        regularization, kernel_dict)
    tf, tt, r, kdict = io.read_train_data('train_data')
    assert np.allclose(train_features, tf) and np.allclose(train_targets, tt)
    assert r == regularization
    for i in kernel_dict:
        for j in kernel_dict[i]:
            if type(kernel_dict[i][j]) != list and \
                    type(kernel_dict[i][j]) != np.ndarray:
                assert kdict[i][j] == kernel_dict[i][j]
            else:
                assert np.allclose(kdict[i][j], kernel_dict[i][j])
    os.remove('{}/train_data.hdf5'.format(wkdir))


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    train_features, train_targets, test_features, test_targets = get_data()
    model = train_model(train_features, train_targets)
    original = test_model(model, test_features, test_targets)
    test_load(original, test_features, test_targets)
    test_raw(train_features, train_targets, model.regularization,
             model.kernel_dict)

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
