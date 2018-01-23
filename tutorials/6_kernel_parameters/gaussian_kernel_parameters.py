"""Tutorial to show impact of changing gaussian kernel parameters."""
import numpy as np
import matplotlib.pyplot as plt

from atoml.regression import GaussianProcess
from atoml.utilities.cost_function import get_error


def afunc(x):
    """Define some polynomial function."""
    y = x - 50.
    p = (y + 4) * (y + 4) * (y + 1) * (y - 1) * (y - 3.5) * (y - 2) * (y - 1)
    p += 40. * y + 80. * np.sin(10. * x)
    return 1. / 20. * p + 500


def plot(sub, prediction):
    """Plotting function."""
    ax = fig.add_subplot(sub)
    ax.plot(linex, liney, '-', lw=1, color='black')
    ax.plot(train, target, 'o', alpha=0.5, color='black')
    ax.plot(test, prediction, 'r-', lw=1, alpha=0.8)
    plt.xlabel('Descriptor')
    plt.ylabel('Response')
    plt.axis('tight')


# A number of training points in x.
train_points = 17
noise_magnitude = 1.

# Randomly generate the training datapoints x.
train = 7.6 * np.random.sample((train_points, 1)) - 4.2 + 50
# Each element in the list train can be referred to as a fingerprint.
# Call the underlying function to produce the target values.
target = np.array(afunc(train))

# Add random noise from a normal distribution to the target values.
target += noise_magnitude * np.random.randn(train_points, 1)

# Generate test datapoints x.
test_points = 513
test = np.vstack(np.linspace(np.min(train) - 0.1, np.max(train) + 0.1,
                             test_points))

# Store the known underlying function for plotting.
linex = np.linspace(np.min(test), np.max(test), test_points)
liney = afunc(linex)

fig = plt.figure(figsize=(20, 10))

for w, p in zip([1.5, 1., 0.5, 0.1], [141, 142, 143, 144]):
    kdict = {'k1': {'type': 'gaussian', 'width': w, 'scaling': 1.}}
    # Set up the prediction routine.
    gp = GaussianProcess(kernel_dict=kdict, regularization=1e-3,
                         train_fp=train,
                         train_target=target,
                         optimize_hyperparameters=False, scale_data=True)
    # Do predictions.
    fit = gp.predict(test_fp=test)

    # Get average errors.
    error = get_error(fit['prediction'], afunc(test))
    print('Gaussian regression error with {0} width: {1:.3f}'.format(
        w, error['absolute_average']))

    # Plotting.
    plot(p, fit['prediction'])

fig = plt.figure(figsize=(20, 10))

for r, p in zip([1., 1e-2, 1e-4, 1e-6], [141, 142, 143, 144]):
    kdict = {'k1': {'type': 'gaussian', 'width': 0.5, 'scaling': 1.}}
    # Set up the prediction routine.
    gp = GaussianProcess(kernel_dict=kdict, regularization=r,
                         train_fp=train,
                         train_target=target,
                         optimize_hyperparameters=False, scale_data=True)
    # Do predictions.
    fit = gp.predict(test_fp=test)

    # Get average errors.
    error = get_error(fit['prediction'], afunc(test))
    print('Gaussian regression error with {0} regularization: {1:.3f}'.format(
        r, error['absolute_average']))

    # Plotting.
    plot(p, fit['prediction'])

fig = plt.figure(figsize=(20, 10))

for s, p in zip([1., 1e2, 1e4, 1e6], [141, 142, 143, 144]):
    kdict = {'k1': {'type': 'gaussian', 'width': 0.5, 'scaling': s}}
    # Set up the prediction routine.
    gp = GaussianProcess(kernel_dict=kdict, regularization=1e-3,
                         train_fp=train,
                         train_target=target,
                         optimize_hyperparameters=False, scale_data=True)
    # Do predictions.
    fit = gp.predict(test_fp=test)

    # Get average errors.
    error = get_error(fit['prediction'], afunc(test))
    print('Gaussian regression error with {0} regularization: {1:.3f}'.format(
        s, error['absolute_average']))

    # Plotting.
    plot(p, fit['prediction'])

fig = plt.figure(figsize=(20, 10))

kdict = {'k1': {'type': 'gaussian', 'width': 0.5, 'scaling': 1.}}
# Set up the prediction routine.
gp = GaussianProcess(kernel_dict=kdict, regularization=1e-3,
                     train_fp=train,
                     train_target=target,
                     optimize_hyperparameters=True, scale_data=True)
# Do predictions.
fit = gp.predict(test_fp=test)

# Get average errors.
error = get_error(fit['prediction'], afunc(test))
print('Gaussian regression error: {0:.3f}'.format(
    error['absolute_average']))

# Plotting.
plot(p, fit['prediction'])

print('Optimized width: {0:.3f}'.format(gp.kernel_dict['k1']['width'][0]))
print('Optimized scale: {0:.3f}'.format(gp.kernel_dict['k1']['scaling']))
print('Optimized regularization: {0:.3f}'.format(gp.regularization))

plt.show()
