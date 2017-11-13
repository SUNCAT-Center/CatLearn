# Toy Model

This tutorial is intended to help you get familiar with using AtoML to set
up a model and do predictions.

First we set up a known underlying function in one dimension. Then we use it to
generate some training data, adding a bit of random noise from a guassian distribution.
Finally we will use AtoML to make predictions on some unseen fingerprint and
benchmark those predictions against the known underlying function.

## Setting up a Gaussian process

AtoML's built-in Gaussian process can be set up like so:

    gp = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                         train_fp=std['train'], train_target=train_targets['target'])

and stored in a variable `gp`. `train_fp` accepts a N by D matrix, where N is the number of training datapoints and D is the number of descriptors. `train_target` accepts a list of N target values. `regularization` (float) is the noise parameter. `kernel_dict` accepts a dictionary defining the kernel. This is defined as:

    kdict = {'k1': {'type': 'gaussian', 'width': w1}}

In this example, a squared exponential kernel aka. gaussian kernel has been chosen by setting the `type` key to `gaussian`.
`width` contains the starting guess for the length scale of the gaussian kernel. `width` can be either a list of the same length as the number of descriptors, or it can be a single float.
The gaussian process optimises all the hyperparameters including the widths and `regularization` if you pass `optimize_hyperparameters=True` to `GaussianProcess`. In the first two examples in the script, however, the optimzer is turned off in order to show you what happens when the model overfits or is too biased.

## Example 1 - biased model.
A biased model not only fits the function poorly. It also underpredicts the uncertainty, because the uncertainty is standard deviation on the distribution of models that can be produced by the process.

## Example 2 - overfitting
The model is predicting the mean, when it is much farther from all datapoints that the characteristic length scale.

