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
                         train_fp=std['train'], train_target=target[0])

and stored in a variable `gp`. Beforehand we have defined a dictionary `kdict` defining the kernel.

    kdict = {'k1': {'type': 'gaussian', 'width': w1}}

In this example, a squared exponential kernel aka. gaussian kernel has been chosen by setting the `type` key to `gaussian`.
`width` contains the starting guess for the length scale of the gaussian kernel.

