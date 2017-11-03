""" This tutorial is intended to help you get familiar with using AtoML to set
up a model and do predictions.

First we set up a known underlying function in one dimension. Then we use it to
generate some training data, adding a bit of random noise.
Finally we will use AtoML to make predictions on some unseen fingerprint and
benchmark those predictions against the known underlying function.
"""
import numpy as np
import matplotlib.pyplot as plt
from atoml.preprocess.feature_preprocess import standardize
from atoml.regression import GaussianProcess


# A known underlying function in one dimension.
def afunc(x):
    """Define some polynomial function."""
    y = x - 50
    p = (y + 4) * (y + 4) * (y + 1) * (y - 1) * (y - 3.5) * (y - 2) * (y - 1)
    p += 40 * y + 80 * np.sin(10 * x)
    return 1. / 20. * p + 500

# Setting up data.
# A number of training points in x.
train_points = 30

# Randomly generate the training datapoints x.
train = 7.6 * np.random.random_sample((1, train_points)) - 4.2 + 50
# Each element in the list train can be referred to as a fingerprint.
# Call the underlying function to produce the target values.
target = afunc(train)

# Add random noise from a normal distribution to the target values.
nt = []
for i in range(train_points):
    nt.append(1.0*np.random.normal())
target += np.array(nt)

# Generate test datapoints x.
test_points = 1001
test = np.linspace(45, 55, test_points)

# Store standard deviations of the training data and targets.
stdx = np.std(train)
stdy = np.std(target)
tstd = np.std(target, axis=1)
# Standardize the training and test data on the same scale.
std = standardize(train_matrix=np.reshape(train, (np.shape(train)[1], 1)),
                  test_matrix=np.reshape(test, (test_points, 1)))

# Model example 1 - biased model.
# Define prediction parameters.
sdt1 = np.sqrt(1e-1)
w1 = 1.0  # Too large widths results in a biased model.
kdict = {'k1': {'type': 'gaussian', 'width': w1}}
# Set up the prediction routine.
gp = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                     train_fp=std['train'], train_target=target[0],
                     optimize_hyperparameters=False)
# Do predictions.
under_fit = gp.predict(test_fp=std['test'], uncertainty=True)
# Get confidence interval on predictions.
upper = np.array(under_fit['prediction']) + \
 (np.array(under_fit['uncertainty'] * tstd))
lower = np.array(under_fit['prediction']) - \
 (np.array(under_fit['uncertainty'] * tstd))

# Model example 2 - over-fitting.
# Define prediction parameters
sdt2 = np.sqrt(1e-5)
w2 = 0.1  # Too small widths lead to over-fitting.
kdict = {'k1': {'type': 'gaussian', 'width': w2}}
# Set up the prediction routine.
gp = GaussianProcess(kernel_dict=kdict, regularization=sdt2**2,
                     train_fp=std['train'], train_target=target[0],
                     optimize_hyperparameters=False)
# Do predictions.
over_fit = gp.predict(test_fp=std['test'], uncertainty=True)
# Get confidence interval on predictions.
over_upper = np.array(over_fit['prediction']) + \
 (np.array(over_fit['uncertainty'] * tstd))
over_lower = np.array(over_fit['prediction']) - \
 (np.array(over_fit['uncertainty'] * tstd))

# Model example 3 - Gaussian Process.
# Set up the prediction routine and optimize hyperparameters.
kdict = {'k1': {'type': 'gaussian', 'width': [w1]}}
gp = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                     train_fp=std['train'], train_target=target[0],
                     optimize_hyperparameters=True)
# Do the optimized predictions.
optimized = gp.predict(test_fp=std['test'], uncertainty=True)
# Get confidence interval on predictions.
opt_upper = np.array(optimized['prediction']) + \
 (np.array(optimized['uncertainty'] * tstd))
opt_lower = np.array(optimized['prediction']) - \
 (np.array(optimized['uncertainty'] * tstd))

# Plotting.
# Store the known underlying function for plotting.
linex = np.linspace(np.min(train), np.max(train), test_points)
liney = afunc(linex)

fig = plt.figure(figsize=(15, 8))

# Example 1
ax = fig.add_subplot(221)
ax.plot(linex, liney, '-', lw=1, color='black')
ax.plot(train[0], target[0], 'o', alpha=0.2, color='black')
ax.plot(test, under_fit['prediction'], 'b-', lw=1, alpha=0.4)
ax.fill_between(test, upper, lower, interpolate=True, color='blue',
                alpha=0.2)
plt.title('Biased kernel regression model.  \n' +
          'w: {0:.3f}, r: {1:.3f}'.format(w1 * stdx, sdt1 * stdy))
plt.xlabel('Descriptor')
plt.ylabel('Response')
plt.axis('tight')

# Example 2
ax = fig.add_subplot(222)
ax.plot(linex, liney, '-', lw=1, color='black')
ax.plot(train[0], target[0], 'o', alpha=0.2, color='black')
ax.plot(test, over_fit['prediction'], 'r-', lw=1, alpha=0.4)
ax.fill_between(test, over_upper, over_lower, interpolate=True, color='red',
                alpha=0.2)
plt.title('Over-fitting kernel regression. \n' +
          'w: {0:.3f}, r: {1:.3f}'.format(w2 * stdx, sdt2 * stdy))
plt.xlabel('Descriptor')
plt.ylabel('Response')
plt.axis('tight')

# Example 3
ax = fig.add_subplot(223)
ax.plot(linex, liney, '-', lw=1, color='black')
ax.plot(train[0], target[0], 'o', alpha=0.2, color='black')
ax.plot(test, optimized['prediction'], 'g-', lw=1, alpha=0.4)
ax.fill_between(test, opt_upper, opt_lower, interpolate=True, color='green',
                alpha=0.2)
plt.title('Optimized GP. \n w: {0:.3f}, r: {1:.3f}'.format(
    gp.kernel_dict['k1']['width'][0]*stdx, np.sqrt(gp.regularization)*stdy))
plt.xlabel('Descriptor')
plt.ylabel('Response')
plt.axis('tight')

# Uncertainty profile.
ax = fig.add_subplot(224)
ax.plot(test, np.array(under_fit['uncertainty'] * tstd), '-', lw=1,
        color='blue')
ax.plot(test, np.array(over_fit['uncertainty'] * tstd), '-', lw=1,
        color='red')
ax.plot(test, np.array(optimized['uncertainty'] * tstd), '-', lw=1,
        color='green')
plt.title('Uncertainty Profiles')
plt.xlabel('Descriptor')
plt.ylabel('Uncertainty')
plt.axis('tight')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.4, hspace=0.4)
plt.show()
