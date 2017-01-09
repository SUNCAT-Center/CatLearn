# AtoML
![Build Status](https://gitlab.com/atoML/AtoML/badges/master/build.svg)
---

Utilities for building and testing Atomic Machine Learning (AtoML) models.
Kernel Ridge Regression (KRR) machine learning routines are implemented. These
will take a numpy array of training and test fingerprint vectors along with a
vector of target values.

In general, any data prepared in this fashion can be fed to the KRR routines,
a number of additional routines have been added that interface with ASE. This
integration allows for the manipulation of atoms objects through KRR
predictions, as well as dynamic generation of fingerprint descriptors through
use of the many ASE functions.

## Requirements

*   [Python](https://www.python.org) 2.6, 2.7, 3.4, 3.5
*   [Numpy](https://docs.scipy.org/doc/numpy/reference/)
*   [ASE](https://wiki.fysik.dtu.dk/ase/) (Atomic Simulation Environment)

## Installation

Put the `<install_dir>/` into your `$PYTHONPATH` environment variable.

#### Atomic Simulation Environment

ASE is a set of tools and Python modules for setting up, manipulating,
running, visualizing and analyzing atomistic simulations. Installation
instructions and tutorials can be found on their webpage
<https://wiki.fysik.dtu.dk/ase/>.

#### ASAP - As Soon As Possible

If a large amount of testing is likely to be performed it is recommended that
ASAP is installed to replace the empirical potentials distributed with ASE. The
ASAP potentials are significantly more computationally efficient.
Installation instructions and tutorials can be found on their webpage
<https://wiki.fysik.dtu.dk/asap>.
