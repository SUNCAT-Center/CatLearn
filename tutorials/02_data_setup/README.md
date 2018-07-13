# Data setup

This tutorial shows how to set up your training data. All prediction functions accept training data in the form of a N x D matrix, where N is a number of training examples and D is the number of descriptors. Each row in the matrix, we call the fingerprint of a training example. Each column, we call a feature.

CatLearn contains functionality to create fingerprints from ase atoms objects. This functionality is done by one or several of the fingerprint generators in CatLearn.
