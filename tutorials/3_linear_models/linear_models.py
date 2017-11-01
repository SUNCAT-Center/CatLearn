""" This tutorial is intended to give further intuition for Gaussian processes.

Like in tutorial 1, we set up a known underlying function, generate training
and test data and calculate predictions and errors. We will compare the results
of linear ridge regression, Gaussian linear kernel regression and finally a
Gaussian process with the usual squared exponential kernel.
"""
import numpy as np
import matplotlib.pyplot as plt
from atoml.feature_preprocess import standardize
from atoml.preprocess.scale_target import target_standardize
from atoml.regression import GaussianProcess, RidgeRegression
from mpl_toolkits.mplot3d import Axes3D


# A known underlying function in two dimensions
def afunc(x):
    """ 2D linear function (plane) """
    return 3. * x[:, 0] - 1. * x[:, 1] + 50.

# Setting up data.
# A number of training points in x.
train_points = 30

# Randomly generate the training datapoints x.
train_d1 = 2 * (np.random.random_sample(train_points) - 0.5)
train_d2 = 2 * (np.random.random_sample(train_points) - 0.5)
train_x1, train_x2 = np.meshgrid(train_d1, train_d2)
train = np.hstack([np.vstack(train_d1), np.vstack(train_d2)])

# Each element in the list train can be referred to as a fingerprint.
# Call the underlying function to produce the target values.
target = np.array(afunc(train))

# Add random noise from a normal distribution to the target values.
for i in range(train_points):
    target[i] += 0.3*np.random.normal()

# Generate test datapoints x.
test_points = 11
test1d = np.vstack(np.linspace(-1.3, 1.3, test_points))
test_x1, test_x2 = np.meshgrid(test1d, test1d)
test = np.hstack([np.vstack(test_x1.ravel()), np.vstack(test_x2.ravel())])

print(np.shape(train))
print(np.shape(test))
print(np.shape(target))

# Store standard deviations of the training data and targets.
stdx = np.std(train)
stdy = np.std(target)
tstd = 1
# Standardize the training and test data on the same scale.
std = standardize(train_matrix=train,
                  test_matrix=test)

# Model example 1 - Ridge regression.
if True:
    # Test ridge regression predictions.
    target_std = target_standardize(target)
    rr = RidgeRegression()
    reg = rr.find_optimal_regularization(X=std['train'],
                                         Y=target_std['target'])
    coef = rr.RR(X=std['train'], Y=target_std['target'], omega2=reg)[0]
    # Test the model.
    sumd = 0.
    rr_predictions = []
    for tf, tt in zip(test, afunc(test)):
        p = ((np.dot(coef, tf)) * target_std['std']) + target_std['mean']
        rr_predictions.append(p)
        sumd += (p - tt) ** 2
    print('Ridge regression prediction:', (sumd / len(test)) ** 0.5)
sdt1 = np.sqrt(1e-6)

if True:
    # Model example 2 - Gausian linear kernel regression.
    # Define prediction parameters
    kdict = {'k1': {'type': 'linear', 'scaling': 1., 'const': 0}}
    # Set up the prediction routine.
    gp1 = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                          train_fp=std['train'], train_target=target,
                          optimize_hyperparameters=True)
    # Do predictions.
    linear = gp1.predict(test_fp=std['test'], uncertainty=True)
    # Get confidence interval on predictions.
    over_upper = np.array(linear['prediction']) + \
     (np.array(linear['uncertainty'] * tstd))
    over_lower = np.array(linear['prediction']) - \
     (np.array(linear['uncertainty'] * tstd))

if True:
    # Model example 3 - Gaussian Process with sqe kernel.
    # Set up the prediction routine and optimize hyperparameters.
    kdict = {'k1': {'type': 'gaussian', 'width': [0.3, 3.]}}
    gp2 = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                          train_fp=std['train'], train_target=target,
                          optimize_hyperparameters=True)
    # Do the optimized predictions.
    optimized = gp2.predict(test_fp=std['test'], uncertainty=True)
    # Get confidence interval on predictions.
    opt_upper = np.array(optimized['prediction']) + \
     (np.array(optimized['uncertainty'] * tstd))
    opt_lower = np.array(optimized['prediction']) - \
     (np.array(optimized['uncertainty'] * tstd))

# Plotting.
# Store the known underlying function for plotting.
#linex1 = np.linspace(np.min(train), np.max(train), test_points)
#liney = afunc(np.hstack([np.vstack(linex1), np.vstack(linex1)]))

plt3d = plt.figure().gca(projection='3d')
plt3d.scatter(train[:, 0], train[:, 1], target,  color='green')

# Example 1 - Ridge regression.
plt3d.plot_surface(test_x1, test_x2,
                   np.array(rr_predictions).reshape(np.shape(test_x1)),
                   alpha=0.3, color='r')
# Example 2 - Gaussian linear regression.
plt3d.plot_surface(test_x1, test_x2,
                   afunc(test).reshape(np.shape(test_x1)),
                   alpha=0.3, color='b')
# Example 3 - Gaussian squared exponential regression.
plt3d.plot_surface(test_x1, test_x2,
                   linear['prediction'].reshape(np.shape(test_x1)),
                   alpha=0.3, color='g')
plt.xlabel('Descriptor 0')
plt.ylabel('Descriptor 1')
plt.axis('tight')
plt.show()
