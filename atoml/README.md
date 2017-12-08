# AtoML Source Code

AtoML is a code base for performing Gaussian Process machine learning on
atomic systems. The code is modular in nature and each module has its own
README, that will provide a more detailed description of what it does.

In general, there are modules for:

-   [Feature generation](#feature-generation)
-   [Preprocessing](#preprocessing)
-   [Regression](#regression)
-   [Cross-validation](#cross-validation)
-   [Utilities](#utilities)

## Feature generation
[(Back to top)](#atoml-source-code)

There are various fingerprint generators available. These typically take a list
of [ASE](https://wiki.fysik.dtu.dk/ase/) atoms object and return an array of
features. The setup functions wrap around some predefined, or user written
generators for various systems. The predefined functions are:

*   adsorbate_fingerprint.py
*   particle_fingerprint.py
*   neighborhood_matrix.py
*   standard_fingerprint.py
*   general_fingerprint.py

## Preprocessing
[(Back to top)](#atoml-source-code)

The module contains functions to scale and optimize the feature space. The
optimization routines include functions that will expand the space with various
transforms and also reduce the space to form more compact representations with
either elimination or extraction.

## Regression
[(Back to top)](#atoml-source-code)

Ridge regression functions to generate reasonable linear models. This will
typically give a good base level of predictive accuracy upon which to benchmark
the more complex Gaussian process. The Gaussian processes functions are also
located in this module. Along with Gaussian process regression, there are also
functions for model optimization.

## Cross-validation
[(Back to top)](#atoml-source-code)

Model testing functions to assess likely error in the predictions.

## Utilities
[(Back to top)](#atoml-source-code)

General utilities to help build and test the models.
