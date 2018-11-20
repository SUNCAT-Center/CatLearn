# CatLearn

> An environment for atomistic machine learning in Python for applications in catalysis.

[![Build Status](https://travis-ci.org/SUNCAT-Center/CatLearn.svg?branch=master)](https://travis-ci.org/SUNCAT-Center/CatLearn) [![Coverage Status](https://coveralls.io/repos/github/SUNCAT-Center/CatLearn/badge.svg?branch=master)](https://coveralls.io/github/SUNCAT-Center/CatLearn?branch=master) [![Documentation Status](https://readthedocs.org/projects/catlearn/badge/?version=latest)](http://catlearn.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/CatLearn.svg)](https://badge.fury.io/py/CatLearn) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Utilities for building and testing atomic machine learning models. Gaussian Processes (GP) regression machine learning routines are implemented. These will take any numpy array of training and test feature matrices along with a vector of target values.

In general, any data prepared in this fashion can be fed to the GP routines, a number of additional functions have been added that interface with [ASE](https://wiki.fysik.dtu.dk/ase/). This integration allows for the manipulation of atoms objects through GP predictions, as well as dynamic generation of descriptors through use of the many ASE functions.

Please see the [tutorials](https://github.com/SUNCAT-Center/CatLearn/tree/master/tutorials) for a detailed overview of what the code can do and the conventions used in setting up the predictive models. For an overview of all the functionality available, please read the [documentation](http://catlearn.readthedocs.io/en/latest/).

## Table of contents

-   [Installation](#installation)
-   [Usage](#usage)
-   [Tutorials](#tutorials)
-   [Functionality](#functionality)
-   [Contribution](#contribution)

## Installation

[(Back to top)](#table-of-contents)

The easiest way to install the code is with:

```shell
$ pip install catlearn
```

This will automatically install the code as well as the dependencies. Alternatively, you can clone the repository to a local directory with:

```shell
$ git clone https://github.com/SUNCAT-Center/CatLearn.git
```

And then put the `<install_dir>/` into your `$PYTHONPATH` environment variable.

Be sure to install dependencies in with:

```shell
$ pip install -r requirements.txt
```

### Docker

To use the docker image, it is necessary to have [docker](https://www.docker.com) installed and running. After cloning the project, build and run the image as follows:

```shell
$ docker build -t catlearn .
```

Then it is possible to use the image in two ways. It is possible to run the docker image as a bash environment in which CatLearn can be used will all dependencies in place.

```shell
$ docker run -it catlearn bash
```

Or python can be run from the docker image.

```shell
$ docker run -it catlearn python2 [file.py]
$ docker run -it catlearn python3 [file.py]
```

Use Ctrl + d to exit the docker image when done.

### Optional Dependencies

The tutorial scripts will generally output some graphical representations of the results etc. For these scripts, it is advisable to have at least `matplotlib` installed:

```shell
$ pip install matplotlib seaborn
```

## Usage

[(Back to top)](#table-of-contents)

In the most basic form, it is possible to set up a GP model and make some predictions using the following lines of code:

```python
import numpy as np
from catlearn.regression import GaussianProcess

# Define some input data.
train_features = np.arange(200).reshape(50, 4)
target = np.random.random_sample((50,))
test_features = np.arange(100).reshape(25, 4)

# Setup the kernel.
kernel = {'k1': {'type': 'gaussian', 'width': 0.5}}

# Train the GP model.
gp = GaussianProcess(kernel_dict=kernel, regularization=1e-3,
                     train_fp=train_features, train_target=target,
                     optimize_hyperparameters=True)

# Get the predictions.
prediction = gp.predict(test_fp=test_features)
```

## Tutorials

[(Back to top)](#table-of-contents)

The above sample of code will train a GP with the squared exponential kernel, fitting some random function. Of course, this isn't so useful, more helpful examples and test scripts are present for most features; primarily, please see the [tutorials](https://github.com/SUNCAT-Center/CatLearn/tree/master/tutorials).

## Functionality

[(Back to top)](#table-of-contents)

There is much functionality in CatLearn to assist in handling atom data and building optimal models. This includes:

-   API to other codes:
    -   [Atomic simulation environment](https://wiki.fysik.dtu.dk/ase/) API
    -   [Magpie](https://bitbucket.org/wolverton/magpie) API
    -   [NetworkX](https://networkx.github.io/) API
-   Fingerprint generators:
    -   Bulk systems
    -   Support/slab systems
    -   Discrete systems
-   Preprocessing routines:
    -   Data cleaning
    -   Feature elimination
    -   Feature engineering
    -   Feature extraction
    -   Feature scaling
-   Regression methods:
    -   Regularized ridge regression
    -   Gaussian processes regression
-   Cross-validation:
    -   K-fold cv
    -   Ensemble k-fold cv
-   General utilities:
    -   K-means clustering
    -   Neighborlist generators
    -   Penalty functions
    -   SQLite db storage

## Contribution

[(Back to top)](#table-of-contents)

Anyone is welcome to contribute to the project. Please see the contribution guide for help setting up a local copy of the code. There are some `TODO` items in the README files for the various modules that give suggestions on parts of the code that could be improved.
