# Regression functions

The folder contains base regression functions.

-   Gaussian processes regression
-   Ridge regression
-   Wrapper for scikit-learn regression

## Gaussian processes regression

Base Gaussian processes regression class. Takes training data and generates a
model, if `optimize_hyperparameters=True`, then the model parameters are
optimized for the supplied data. Call the `predict` function to then make
predictions on test data.

## Ridge regression

Base ridge regression class. Takes training data and optimizes regularization
before finding the coefficients. Call the `predict` function to then make
predictions on test data.

## Wrapper

This just provides a wrapper for some scikit-learn regression functions that
are used in some of the preprocessing steps. 
