# Regression functions

The folder contains base regression functions.

-   [Gaussian processes regression](#gaussian-processes-regression)
-   [Ridge regression](#ridge-regression)
-   [Wrapper for scikit-learn regression](#wrapper)

## Gaussian processes regression
[(Back to top)](#regression-functions)

Base Gaussian processes regression class. The class is initialized with some
training data (features and targets), then a model is generated based on some
flags that are set.

  ```python
    gp = GaussianProcess(train_features, train_target, kernel)
  ```

A kernel setup must also always be passed to the gp. The available kernels are
defined in `gpfunctions/kernels.py` but in general have a uniform setup, though
parameters that must be specified differ.

  ```python
    kernel = {'k1': {'type': 'gaussian', 'width': 0.5, 'const': 1.0}}
  ```

It is also possible to append numerous kernel functions together, which can be
applied selectively to different sets of features.

  ```python
    kernel = {'k1': {'type': 'linear', 'features': [0, 1], 'const': 0.},
              'k2': {'type': 'gaussian', 'features': [2, 3], 'width': 0.5,
                     'operation': 'multiplication'}}
  ```

If `optimize_hyperparameters=True`, then the model parameters are optimized for
the supplied data. This is performed based on maximizing the log marginal
likelihood. These can be accessed by recalling the kernel dictionary and
regularization parameter:

  ```python
    gp.kernel_dict, gp.regularization
  ```

Call the `predict` function to then make predictions on test data. A number of
additional flags can be set here to calculate the e.g. variance on the
predicted mean. It is important to note that this is currently *not* rescaled
in any way, even if input features are scaled.

  ```python
    p = gp.predict(test, uncertainty=True)
    p['uncertainty']
    p['prediction']
  ```

## Ridge regression
[(Back to top)](#regression-functions)

Base ridge regression class. The class can be initialized taking only the
default setup.

  ```python
    rr = RidgeRegression()
  ```

Call the `predict` function to then make predictions on test data. Takes
training data (features and targets) and optimizes regularization before
finding coefficients. The optimal model is then applied to the test data and
predictions returned.

  ```python
    rr.predict(train_features, train_target, test_features)
  ```

## Wrapper
[(Back to top)](#regression-functions)

This just provides a wrapper for some scikit-learn regression functions that
are used in some of the preprocessing steps. This probably shouldn't be used
for other purposes as the models being trained are too basic.
