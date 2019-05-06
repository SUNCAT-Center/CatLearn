# CatLearn

> An environment for atomistic machine learning in Python for applications in catalysis.

[![DOI](https://zenodo.org/badge/130307939.svg)](https://zenodo.org/badge/latestdoi/130307939) [![Build Status](https://travis-ci.org/SUNCAT-Center/CatLearn.svg?branch=master)](https://travis-ci.org/SUNCAT-Center/CatLearn) [![Coverage Status](https://coveralls.io/repos/github/SUNCAT-Center/CatLearn/badge.svg?branch=master)](https://coveralls.io/github/SUNCAT-Center/CatLearn?branch=master) [![Documentation Status](https://readthedocs.org/projects/catlearn/badge/?version=latest)](http://catlearn.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/CatLearn.svg)](https://badge.fury.io/py/CatLearn) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Utilities for building and testing atomic machine learning models. Gaussian Processes (GP) regression machine learning routines are implemented. These will take any numpy array of training and test feature matrices along with a vector of target values.

In general, any data prepared in this fashion can be fed to the GP routines, a number of additional functions have been added that interface with [ASE](https://wiki.fysik.dtu.dk/ase/). This integration allows for the manipulation of atoms objects through GP predictions, as well as dynamic generation of descriptors through use of the many ASE functions.

CatLearn also includes the [MLNEB](https://github.com/SUNCAT-Center/CatLearn/tree/master/tutorials/11_NEB) algorithm for efficient transition state search, and the [MLMIN](https://github.com/SUNCAT-Center/CatLearn/tree/master/tutorials/12_MLMin) algorithm for efficient atomic structure optimization.

Please see the [tutorials](https://github.com/SUNCAT-Center/CatLearn/tree/master/tutorials) for a detailed overview of what the code can do and the conventions used in setting up the predictive models. For an overview of all the functionality available, please read the [documentation](http://catlearn.readthedocs.io/en/latest/).

## Table of contents

-   [Installation](#installation)
-   [Tutorials](#tutorials)
-   [Usage](#usage)
-   [Functionality](#functionality)
-   [How to cite](#how-to-cite-catlearn)
-   [Contribution](#contribution)

## Installation

[(Back to top)](#table-of-contents)

The easiest way to install the code is with:

```shell
$ pip install catlearn
```

This will automatically install the code as well as the dependencies. 

### Installation without dependencies

[(Back to top)](#table-of-contents)

If you want to install catlearn without dependencies, you can do:

```shell
$ pip install catlearn --no-deps
```

MLMIN and MLNEB will not need anything apart from ASE 3.17.0 or newer to run, but there are other parts of the code, which need the dependencies listed in [requirements.txt](https://github.com/SUNCAT-Center/CatLearn/blob/master/requirements.txt)

### Developer installation

```shell
$ git clone https://github.com/SUNCAT-Center/CatLearn.git
```

And then put the `<install_dir>/` into your `$PYTHONPATH` environment variable.

You can install dependencies in with:

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

## Tutorials

[(Back to top)](#table-of-contents)

Helpful examples and test scripts are present in [tutorials](https://github.com/SUNCAT-Center/CatLearn/tree/master/tutorials).

## Usage

[(Back to top)](#table-of-contents)

Set up CatLearn's Gaussian Process model and make some predictions using the following lines of code:

```python
import numpy as np
from catlearn.regression import GaussianProcess

# Define some input data.
train_features = np.arange(200).reshape(50, 4)
target = np.random.random_sample((50,))
test_features = np.arange(100).reshape(25, 4)

# Setup the kernel.
kernel = [{'type': 'gaussian', 'width': 0.5}]

# Train the GP model.
gp = GaussianProcess(kernel_list=kernel, regularization=1e-3,
                     train_fp=train_features, train_target=target,
                     optimize_hyperparameters=True)

# Get the predictions.
prediction = gp.predict(test_fp=test_features)
```

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
-   Machine Learning Algorithms
    -   Machine Learning Nudged Elastic Band (ML-NEB) algorithm.
-   General utilities:
    -   K-means clustering
    -   Neighborlist generators
    -   Penalty functions
    -   SQLite db storage

## How to cite CatLearn

[(Back to top)](#table-of-contents)

If you find CatLearn useful in your research, please cite

    1) M. H. Hansen, J. A. Garrido Torres, P. C. Jennings, 
       Z. Wang, J. R. Boes, O. G. Mamun and T. Bligaard.
       An Atomistic Machine Learning Package for Surface Science and Catalysis.
       https://arxiv.org/abs/1904.00904

If you use CatLearn's ML-NEB module, please cite:

    2) J. A. Garrido Torres, M. H. Hansen, P. C. Jennings,
       J. R. Boes and T. Bligaard. Phys. Rev. Lett. 122, 156001.
       https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.156001

## Contribution

[(Back to top)](#table-of-contents)

Anyone is welcome to contribute to the project. Please see the contribution guide for help setting up a local copy of the code. There are some `TODO` items in the README files for the various modules that give suggestions on parts of the code that could be improved.
