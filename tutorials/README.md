# AtoML Tutorials

Here are a few tutorials to help setup some predictive functions and assess
the resulting models. These are kept relatively simple, with data only spanning
a small number of dimensions.

## Tutorials

-   [Toy Model](#toy-model)
-   [Data Setup](#data-setup)
-   [Linear Models](#linear-models)
-   [Uncertainties](#uncertainties)
-   [Toy Model Gradients](#gradients)

## Toy Model
[(Back to top)](#atoml-tutorials)

Set up a known underlying function in one dimension. Then generate some
training data, adding a bit of random noise from a Gaussian distribution.
Finally use AtoML to make predictions on some unseen data and benchmark those
predictions against the known underlying function.

## Data Setup
[(Back to top)](#atoml-tutorials)

AtoML contains functionality to create fingerprints from
[ASE](https://wiki.fysik.dtu.dk/ase/) atoms objects. This functionality is
achieved with one or several of the fingerprint generators in AtoML. In this
tutorial the adsorbate fingerprint generator is utilized, which is useful for
converting adsorbates on extended surfaces into fingerprints for predicting
their chemisorption energies.

## Linear Models
[(Back to top)](#atoml-tutorials)

This tutorial is intended to give further intuition for Gaussian processes.
Results are compared for linear ridge regression, Gaussian linear kernel
regression and finally a Gaussian process with the popular squared exponential
kernel.

## Uncertainties
[(Back to top)](#atoml-tutorials)

Set up the same known underlying function as in the previous tutorial, generate
training and test data and calculate predictions and uncertainty estimates.
Compare error intervals of Gaussian linear kernel regression and the Gaussian
process with the popular squared exponential kernel.

## Gradients
[(Back to top)](#atoml-tutorials)

Setup a known function and generate some training data with the gradients
information. Make predictions and compare how well the Gaussian process
performs with and without the gradients information included.
