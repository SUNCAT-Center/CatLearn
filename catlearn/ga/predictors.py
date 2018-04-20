"""Some generic prediction functions."""
import time

from catlearn.regression import GaussianProcess


def minimize_error(train_features, train_targets, test_features, test_targets):
    """A generic fitness function.

    This fitness function will minimize the cost function.

    Parameters
    ----------
    train_features : array
        The training features.
    train_targets : array
        The training targets.
    test_features : array
        The test feaatures.
    test_targets : array
        The test targets.
    """
    kernel = {'k1': {'type': 'gaussian', 'width': 1.,
                     'scaling': 1., 'dimension': 'single'}}

    gp = GaussianProcess(train_fp=train_features,
                         train_target=train_targets,
                         kernel_dict=kernel,
                         regularization=1e-2,
                         optimize_hyperparameters=True,
                         scale_data=True)

    pred = gp.predict(test_fp=test_features, test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)

    score = pred['validation_error']['rmse_average']

    return [-score]


def minimize_error_descriptors(train_features, train_targets, test_features,
                               test_targets):
    """A generic fitness function.

    This fitness function will minimize the cost function as well as the number
    of descriptors. This will provide a Pareto optimial set of solutions upon
    convergence.

    Parameters
    ----------
    train_features : array
        The training features.
    train_targets : array
        The training targets.
    test_features : array
        The test feaatures.
    test_targets : array
        The test targets.
    """
    kernel = {'k1': {'type': 'gaussian', 'width': 1.,
                     'scaling': 1., 'dimension': 'single'}}

    gp = GaussianProcess(train_fp=train_features,
                         train_target=train_targets,
                         kernel_dict=kernel,
                         regularization=1e-2,
                         optimize_hyperparameters=True,
                         scale_data=True)

    pred = gp.predict(test_fp=test_features, test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)

    score = pred['validation_error']['rmse_average']
    dimension = train_features.shape[1]

    return [-score, -dimension]


def minimize_error_time(train_features, train_targets, test_features,
                        test_targets):
    """A generic fitness function.

    This fitness function will minimize the cost function as well as the time
    to train the model. This will provide a Pareto optimial set of solutions
    upon convergence.

    Parameters
    ----------
    train_features : array
        The training features.
    train_targets : array
        The training targets.
    test_features : array
        The test feaatures.
    test_targets : array
        The test targets.
    """
    kernel = {'k1': {'type': 'gaussian', 'width': 1.,
                     'scaling': 1., 'dimension': 'single'}}

    stime = time.time()
    gp = GaussianProcess(train_fp=train_features,
                         train_target=train_targets,
                         kernel_dict=kernel,
                         regularization=1e-2,
                         optimize_hyperparameters=True,
                         scale_data=True)
    timing = time.time() - stime

    pred = gp.predict(test_fp=test_features, test_target=test_targets,
                      get_validation_error=True,
                      get_training_error=True)

    score = pred['validation_error']['rmse_average']

    return [-score, -timing]
