# AtoML
[![pipeline status](https://gitlab.com/atoML/AtoML/badges/master/pipeline.svg)](https://gitlab.com/atoML/AtoML/commits/master)
[![coverage report](https://gitlab.com/atoML/AtoML/badges/master/coverage.svg)](https://gitlab.com/atoML/AtoML/commits/master)

Utilities for building and testing Atomic Machine Learning (AtoML) models.
Gaussian Processes (GP) regression machine learning routines are implemented.
These will take any numpy array of training and test feature matrices along
with a vector of target values.

In general, any data prepared in this fashion can be fed to the GP routines,
a number of additional functions have been added that interface with
[ASE](https://wiki.fysik.dtu.dk/ase/). This integration allows for the
manipulation of atoms objects through GP predictions, as well as dynamic
generation of descriptors through use of the many ASE functions.

## Table of contents

-   [Installation](#installation)
-   [Usage](#usage)
-   [Contribution](#contribution)
-   [Authors](#authors)

## Installation
[(Back to top)](#table-of-contents)

The easiest way to install the code is with:

  ```shell
    pip install git+https://gitlab.com/atoML/AtoML.git
  ```

This will automatically install the code as well as the dependencies.
Alternatively, you can clone the repository to a local directory with:

  ```shell
    git clone https://gitlab.com/atoML/AtoML.git
  ```

And then put the `<install_dir>/` into your `$PYTHONPATH` environment variable.

Be sure to install dependencies in with:

  ```shell
    pip install -r requirements.txt
  ```

#### Docker
It is necessary to have [docker](https://www.docker.com) installed and running.
Then it is possible to run AtoML inside a docker image. After cloning the
project, build and run the image as follows:

  ```shell
    docker build -t atoml .
    docker run -it atoml bash
  ```

Use ctrl + d to exit the docker image when done.

#### Optional Dependencies

The tutorial scripts will generally output some graphical representations of
the results etc. For these scripts, it is advisable to have at least
`matplotlib` installed:

  ```shell
    pip install matplotlib pandas seaborn
  ```

## Usage
[(Back to top)](#table-of-contents)

In the most basic form, it is possible to set up a GP model and make some
predictions using the following lines of code:

  ```python
    import numpy as np
    from atoml.regression import GaussianProcess

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

The above sample of code will train a GP with the squared exponential kernel,
fitting some random function. Of course this isn't so useful, more helpful
examples and test scripts are present for most features.

## Contribution
[(Back to top)](#table-of-contents)

Anyone is welcome to contribute to the project. Please see the contribution
guide for help setting up a local copy of the code. There are some `TODO` items
in the README files for the various modules that give suggestions on parts of
the code that could be improved.

## Authors
[(Back to top)](#table-of-contents)

#### Lead
-   [Paul Jennings](http://suncat.stanford.edu/theory/people/paul-jennings)
-   [Martin Hansen](https://suncat.stanford.edu/theory/people/martin-hangaard-hansen)
-   [Thomas Bligaard](https://suncat.stanford.edu/theory/people/thomas-bligaard)

#### Contributors
-   Jose A. Garrido Torres
-   Jacob Boes
-   Andrew Doyle
-   Markus Ekvall
