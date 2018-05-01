# CatLearn Tutorials

Here are a few tutorials to help set up some predictive functions and assess the resulting models.

## Tutorials

1.  [Toy Model](#toy-model)
2.  [Data Setup](#data-setup)
3.  [Linear Models](#linear-models)
4.  [Uncertainties](#uncertainties)
5.  [Toy Model Gradients](#gradients)
6.  [Kernel Parameters](#kernel-parameters)
7.  [Cross Validation](#cross-validation)
8.  [Organic Molecules](#organic-molecules)
9.  [Bulk Fingerprints](#bulk-fingerprints)
10. [Feature Selection](#feature-selection)

## Toy Model

[(Back to top)](#atoml-tutorials)

Set up a known underlying function in one dimension. Then generate some training data, adding a bit of random noise from a Gaussian distribution. Finally, use CatLearn to make predictions on some unseen data and benchmark those predictions against the known underlying function. Within this tutorial, there is also a benchmark against the [GPflow](https://github.com/GPflow/GPflow) code (this will need to be installed).

## Data Setup

[(Back to top)](#atoml-tutorials)

CatLearn contains functionality to create fingerprints from [ASE](https://wiki.fysik.dtu.dk/ase/) atoms objects. This functionality is achieved with one or several of the fingerprint generators in CatLearn. In the first of these tutorials, the adsorbate fingerprint generator is utilized, which is useful for converting adsorbates on extended surfaces into fingerprints for predicting their chemisorption energies. Some general atomic properties are generated in the second tutorial. In the third, fingerprints are generated for discrete systems, in this case, some nanoparticle data.

## Linear Models

[(Back to top)](#atoml-tutorials)

This tutorial is intended to give further intuition for Gaussian processes. Results are compared for linear ridge regression, Gaussian linear kernel regression and finally a Gaussian process with the popular squared exponential kernel.

## Uncertainties

[(Back to top)](#atoml-tutorials)

Set up the same known underlying function as in the previous tutorial, generate training and test data and calculate predictions and uncertainty estimates. Compare error intervals of Gaussian linear kernel regression and the Gaussian process with the popular squared exponential kernel.

## Gradients

[(Back to top)](#atoml-tutorials)

Set up a known function and generate some training data with the gradients information. Make predictions and compare how well the Gaussian process performs with and without the gradients information included.

## Kernel Parameters

[(Back to top)](#atoml-tutorials)

In this tutorial, the model hyperparameters are investigated for two popular kernels, the linear and squared exponential. The effects of varying the various kernel parameters, as well as the regularization term on the model are shown.

## Cross Validation

[(Back to top)](#atoml-tutorials)

Tutorials are provided for running both k-fold and hierarchy cross-validation. This is useful for validating the accuracy of a model as well as generating a learning curve in the case of the hierarchy cv.

## Organic Molecules

[(Back to top)](#atoml-tutorials)

This provides a more detailed look at setting up some models based on data downloaded from [CMR](https://cmr.fysik.dtu.dk/). In particular, there is some analysis of the feature space presented in these tutorials.

## Bulk Fingerprints

[(Back to top)](#atoml-tutorials)

A feature space is generated for some bulk compounds.

## Feature Selection

[(Back to top)](#atoml-tutorials)

In these tutorials, feature elimination is investigated based on greedy algorithms as well as a genetic algorithm.
