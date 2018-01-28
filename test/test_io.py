"""Test io functions."""
import os

from atoml.regression import GaussianProcess
from atoml.utilities import io
from common import get_data


def train_model(train_features, train_targets):
    """Function to train a Gaussian process."""
    kdict = {'k1': {'type': 'gaussian', 'width': 1., 'scaling': 1.}}
    gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                         kernel_dict=kdict, regularization=1e-3,
                         optimize_hyperparameters=True, scale_data=True)

    io.write(filename='test-model', model=gp)

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
    gp = io.read(filename='test-model')

    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)

    assert all(pred['validation_error']['rmse_all'] ==
               original['validation_error']['rmse_all'])

    os.remove('test-model.pkl')


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    train_features, train_targets, test_features, test_targets = get_data()
    model = train_model(train_features, train_targets)
    original = test_model(model, test_features, test_targets)
    test_load(original, test_features, test_targets)

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
