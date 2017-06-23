# AtoML
![Build Status](https://gitlab.com/atoML/AtoML/badges/master/build.svg)

Utilities for building and testing Atomic Machine Learning (AtoML) models.
Gaussian Processes (GP) machine learning routines are implemented. These
will take a numpy array of training and test feature matrices along with a
vector of target values.

In general, any data prepared in this fashion can be fed to the GP routines,
a number of additional functions have been added that interface with ASE. This
integration allows for the manipulation of atoms objects through GP
predictions, as well as dynamic generation of descriptors through use of the
many ASE functions.

## AtoML functions

*   Manipulate list of atoms objects to form training and test data. Useful
when getting data from e.g. a database.
    -   data_setup.py
*   Convert ASE atoms objects into feature vectors for a number of potentially
interesting problems.
    -   fingerprint_setup.py
    -   adsorbate_fingerprint.py
    -   particle_fingerprint.py
    -   standard_fingerprint.py
    -   neighborhood_matrix.py
*   Database functions to store the feature matrix from a given dataset.
    -   database_functions.py
*   Feature preprocessing, engineering, elimination and extraction methods.
    -   feature_preprocess.py
    -   feature_engineering.py
    -   feature_elimination.py
    -   feature_extraction.py
*   Gaussian processes predictions with hyperparameter optimization.
    -   predict.py
    -   kernels.py
    -   covarience.py
    -   model_selection.py
    -   build_model.py

## Installation

Put the `<install_dir>/` into your `$PYTHONPATH` environment variable.

## Requirements

*   [Python](https://www.python.org) 2.7, 3.4, 3.5
*   [Numpy](https://docs.scipy.org/doc/numpy/reference/)
*   [ASE](https://wiki.fysik.dtu.dk/ase/) (Atomic Simulation Environment)

## Optional

*   [ASAP](https://wiki.fysik.dtu.dk/asap)
*   [Pandas](http://pandas.pydata.org)
*   [Seaborn](http://seaborn.pydata.org)
*   [scikit-learn](http://scikit-learn.org/stable/)

## Dependencies
##### Atomic Simulation Environment

ASE is a set of tools and Python modules for setting up, manipulating,
running, visualizing and analyzing atomistic simulations. Installation
instructions and tutorials can be found on their webpage
<https://wiki.fysik.dtu.dk/ase/>.

##### scikit-learn

Some scikit-learn functions are utilized within the model optimization
routines. In particular when comparing to linear models, we utilize this
framework extensively. If there is no interest in this level of
optimization/benchmarking, then the code base is not needed.

##### ASAP - As Soon As Possible

If a large amount of testing is likely to be performed it is recommended that
ASAP is installed to replace the empirical potentials distributed with ASE. The
ASAP potentials are significantly more computationally efficient.
Installation instructions and tutorials can be found on their webpage
<https://wiki.fysik.dtu.dk/asap>.

##### Visualization

Data visualization in the example scripts is performed with the
[Pandas](http://pandas.pydata.org) and [Seaborn](http://seaborn.pydata.org)
Python packages.
