"""First CatLearn tutorial.

This tutorial is intended to help you get familiar with using CatLearn to set
up a model and do predictions.

First we set up a known underlying function in one dimension. Then we use it to
generate some training data, adding a bit of random noise.
Finally we will use CatLearn to make predictions on some unseen fingerprint and
benchmark those predictions against the known underlying function.
"""
import numpy as np
import matplotlib.pyplot as plt

from catlearn.preprocess.scaling import standardize, target_standardize
from catlearn.regression import GaussianProcess
from catlearn.regression.cost_function import get_error

# Look for alternative GPs.
try:
    import gpflow
    haz_gpflow = True
except ImportError:
    haz_gpflow = False
try:
    import GPy
    haz_gpy = True
except ImportError:
    haz_gpy = False


# A known underlying function in one dimension.
def afunc(x):
    """Define some polynomial function."""
    y = x - 50.
    p = (y + 4) * (y + 4) * (y + 1) * (y - 1) * (y - 3.5) * (y - 2) * (y - 1)
    p += 40. * y + 80. * np.sin(10. * x)
    return 1. / 20. * p + 500


# Setting up data.
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

# Store standard deviations of the training data and targets.
stdx = np.std(train)
stdy = np.std(target)
tstd = 2.

# Standardize the training and test data on the same scale.
std = standardize(train_matrix=train,
                  test_matrix=test)
# Standardize the training targets.
train_targets = target_standardize(target)
# Note that predictions will now be made on the standardized scale.

# Store the known underlying function for plotting.
linex = np.linspace(np.min(test), np.max(test), test_points)
liney = afunc(linex)
# Plotting.
fig = plt.figure(figsize=(15, 8))
if haz_gpy or haz_gpflow:
    grid = 230
    last = 6
else:
    grid = 220
    last = 4
ax224 = fig.add_subplot(grid+last)
plt.title('Uncertainty Profiles')
plt.xlabel('Descriptor')
plt.ylabel('Uncertainty')

if True:
    # Model example 1 - biased model.
    # Define prediction parameters.
    sdt1 = 0.001
    # Too large width results in a biased model.
    w1 = 3.0
    kdict = [{'type': 'gaussian', 'width': w1}]
    # Set up the prediction routine.
    gp = GaussianProcess(kernel_dict=kdict, regularization=sdt1,
                         train_fp=std['train'],
                         train_target=train_targets['target'],
                         optimize_hyperparameters=False)
    # Do predictions.
    under_fit = gp.predict(test_fp=std['test'], uncertainty=True)
    # Scale predictions back to the original scale.
    under_prediction = np.vstack(under_fit['prediction']) * \
        train_targets['std'] + train_targets['mean']
    under_uncertainty = np.vstack(under_fit['uncertainty']) * \
        train_targets['std']
    # Get average errors.
    error = get_error(under_prediction.reshape(-1), afunc(test).reshape(-1))
    print('Gaussian linear regression prediction:', error['absolute_average'])
    # Get confidence interval on predictions.
    upper = under_prediction + under_uncertainty * tstd
    lower = under_prediction - under_uncertainty * tstd

    # Plot example 1.
    ax = fig.add_subplot(grid+1)
    ax.plot(linex, liney, '-', lw=1, color='black')
    ax.plot(train, target, 'o', alpha=0.2, color='black')
    ax.plot(test, under_prediction, 'b-', lw=1, alpha=0.4)
    ax.fill_between(np.hstack(test), np.hstack(upper), np.hstack(lower),
                    interpolate=True, color='blue',
                    alpha=0.2)
    plt.title('Biased kernel regression model.  \n' +
              'w: {0:.3f}, r: {1:.3f}'.format(w1 * stdx,
                                              sdt1 * stdy))
    plt.xlabel('Descriptor')
    plt.ylabel('Response')
    plt.axis('tight')

if True:
    # Model example 2 - over-fitting.
    # Define prediction parameters
    sdt2 = 0.001
    # Too small width lead to over-fitting.
    w2 = 0.03
    kdict = [{'type': 'gaussian', 'width': w2}]
    # Set up the prediction routine.
    gp = GaussianProcess(kernel_dict=kdict, regularization=sdt2,
                         train_fp=std['train'],
                         train_target=train_targets['target'],
                         optimize_hyperparameters=False)
    # Do predictions.
    over_fit = gp.predict(test_fp=std['test'], uncertainty=True)
    # Scale predictions back to the original scale.
    over_prediction = np.vstack(over_fit['prediction']) * \
        train_targets['std'] + train_targets['mean']
    over_uncertainty = np.vstack(over_fit['uncertainty']) * \
        train_targets['std']
    # Get average errors.
    error = get_error(over_prediction.reshape(-1), afunc(test).reshape(-1))
    print('Gaussian kernel regression prediction:', error['absolute_average'])
    # Get confidence interval on predictions.
    over_upper = over_prediction + over_uncertainty * tstd
    over_lower = over_prediction - over_uncertainty * tstd

    # Uncertainty profile.
    ax224.plot(test, np.array(under_uncertainty * tstd), '-', lw=1,
               color='blue')

    # Plot example 2.
    ax = fig.add_subplot(grid+2)
    ax.plot(linex, liney, '-', lw=1, color='black')
    ax.plot(train, target, 'o', alpha=0.2, color='black')
    ax.plot(test, over_prediction, 'r-', lw=1, alpha=0.4)
    ax.fill_between(np.hstack(test), np.hstack(over_upper),
                    np.hstack(over_lower), interpolate=True, color='red',
                    alpha=0.2)
    plt.title('Over-fitting kernel regression. \n' +
              'w: {0:.3f}, r: {1:.3f}'.format(w2 * stdx,
                                              sdt2 * stdy))
    plt.xlabel('Descriptor')
    plt.ylabel('Response')
    plt.axis('tight')
    # Uncertainty profile.
    ax224.plot(test, np.array(over_uncertainty * tstd), '-', lw=1,
               color='red')

if True:
    # Model example 3 - Gaussian Process.
    # Set up the prediction routine and optimize hyperparameters.
    w3 = 0.1
    sdt3 = 0.001
    kdict = [{'type': 'gaussian', 'width': [w3]}]

    gp = GaussianProcess(kernel_dict=kdict, regularization=sdt3,
                         train_fp=std['train'],
                         train_target=train_targets['target'],
                         optimize_hyperparameters=True)
    print('Optimized kernel:', gp.kernel_dict)
    print(-gp.theta_opt['fun'])
    # Do the optimized predictions.
    optimized = gp.predict(test_fp=std['test'], uncertainty=True)
    # Scale predictions back to the original scale.
    opt_prediction = np.vstack(optimized['prediction']) * \
        train_targets['std'] + train_targets['mean']
    opt_uncertainty = np.vstack(optimized['uncertainty_with_reg']) * \
        train_targets['std']
    # Get average errors.
    error = get_error(opt_prediction.reshape(-1), afunc(test).reshape(-1))
    print('Gaussian kernel regression prediction:', error['absolute_average'])
    # Get confidence interval on predictions.
    opt_upper = opt_prediction + opt_uncertainty * tstd
    opt_lower = opt_prediction - opt_uncertainty * tstd

    # Plot example 3.
    ax = fig.add_subplot(grid+3)
    ax.plot(linex, liney, '-', lw=1, color='black')
    ax.plot(train, target, 'o', alpha=0.2, color='black')
    ax.plot(test, opt_prediction, 'g-', lw=1, alpha=0.4)
    ax.fill_between(np.hstack(test), np.hstack(opt_upper),
                    np.hstack(opt_lower), interpolate=True,
                    color='green', alpha=0.2)
    plt.title('Optimized GP. \n w: {0:.3f}, r: {1:.3f}'.format(
        gp.kernel_dict[0]['width'][0] * stdx,
        np.sqrt(gp.regularization) * stdy))
    plt.xlabel('Descriptor')
    plt.ylabel('Response')
    plt.axis('tight')
    # Uncertainty profile.
    ax224.plot(test, np.array(opt_uncertainty * tstd), '-', lw=1,
               color='green')

if haz_gpflow:
    # Model example 4 - GPflow.
    k = gpflow.kernels.RBF(1, lengthscales=0.1)
    m = gpflow.models.GPR(np.vstack(std['train']), train_targets['target'],
                          kern=k)
    m.likelihood.variance = 0.00003
    gpflow.train.ScipyOptimizer().minimize(m)
    mean, var = m.predict_y(std['test'])
    mean = mean * train_targets['std'] + train_targets['mean']
    std = (var ** 0.5) * train_targets['std']
    opt_upper = mean + std * tstd
    opt_lower = mean - std * tstd
    ax = fig.add_subplot(grid+4)
    ax.plot(linex, liney, '-', lw=1, color='black')
    ax.plot(train, target, 'o', alpha=0.2, color='black')
    ax.plot(test, mean, 'g-', lw=1, alpha=0.4)
    ax.fill_between(np.hstack(test), np.hstack(opt_upper),
                    np.hstack(opt_lower), interpolate=True,
                    color='purple', alpha=0.2)
    plt.title('Optimized GPflow')
    plt.xlabel('Descriptor')
    plt.ylabel('Response')
    plt.axis('tight')
    # Uncertainty profile.
    ax224.plot(test, np.array(std * tstd), '-', lw=1,
               color='purple')

if haz_gpy:
    # Model example 5 - GPy.
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(std['train'], train_targets['target'], kernel)
    m.optimize()
    print(m)
    ax = fig.add_subplot(grid+5)
    ax.plot(linex, liney, '-', lw=1, color='black')
    ax.plot(train, target, 'o', alpha=0.2, color='black')
    ax.plot(test, opt_prediction, 'g-', lw=1, alpha=0.4)
    ax.fill_between(np.hstack(test), np.hstack(opt_upper),
                    np.hstack(opt_lower), interpolate=True,
                    color='brown', alpha=0.2)
    plt.title('Optimized GPy')
    plt.xlabel('Descriptor')
    plt.ylabel('Response')
    plt.axis('tight')
    # Uncertainty profile.
    ax224.plot(test, np.array(opt_uncertainty * tstd), '-', lw=1,
               color='brown')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.4, hspace=0.4)
plt.show()
