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

## Installation

Put the `<install_dir>/` into your `$PYTHONPATH` environment variable.

Install dependencies in with:

    pip install -r requirements.txt

Examples and test scripts are present for most features.

## Contribution

See the contribution guide.
