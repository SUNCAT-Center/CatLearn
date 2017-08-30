# AtoML
![Build Status](https://gitlab.com/atoML/AtoML/badges/master/build.svg)

Utilities for building and testing Atomic Machine Learning (AtoML) models.
Gaussian Processes (GP) regression machine learning routines are implemented.
These will take any numpy array of training and test feature matrices along
with a vector of target values.

In general, any data prepared in this fashion can be fed to the GP routines,
a number of additional functions have been added that interface with
[ASE](https://wiki.fysik.dtu.dk/ase/). This integration allows for the
manipulation of atoms objects through GP predictions, as well as dynamic
generation of descriptors through use of the many ASE functions.

## AtoML functions

*   Manipulate list of atoms objects to form training and test data. Useful
when getting data from e.g. a database.
    -   data_setup.py
*   Convert ASE atoms objects into feature vectors for a number of potentially
interesting problems.
    -   fingerprint_setup.py
    -   adsorbate_fingerprint.py
    -   particle_fingerprint.py
    -   neighborhood_matrix.py
    -   standard_fingerprint.py
    -   general_fingerprint.py
*   Database functions to store the feature matrix from a given dataset.
    -   database_functions.py
*   Feature preprocessing, engineering, elimination and extraction methods.
    -   feature_preprocess.py
    -   feature_engineering.py
    -   feature_elimination.py
    -   feature_extraction.py
*   Gaussian processes predictions with with model optimization.
    -   predict.py
    -   kernels.py
    -   covarience.py
    -   model_selection.py
    -   uncertainty.py
    -   gp_sensitivity.py
    -   cost_function.py
*   Model testing functions.
    -   cross_validation.py

## Installation

Put the `<install_dir>/` into your `$PYTHONPATH` environment variable.

Install dependencies in with:

    pip install -r requirements.txt

Examples and test scripts are present for most features.

## Contribution

See the contribution guide.
