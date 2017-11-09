""" This tutorial is intended to give further intuition for Gaussian processes.

Like in tutorial 1, we set up a known underlying function, generate training
and test data and calculate predictions and errors. We will compare the results
of linear ridge regression, Gaussian linear kernel regression and finally a
Gaussian process with the usual squared exponential kernel.

Try playing around with the training set size (density) and noise_magnitude.

"""
import numpy as np
import matplotlib.pyplot as plt
from atoml.preprocess.feature_preprocess import standardize
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
# Magnitude of the noise.
noise_magnitude = 0.1

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
    target[i] += noise_magnitude * np.random.normal()

# Generate test datapoints x.
test_points = 10
test1d = np.vstack(np.linspace(-1.3, 1.3, test_points))
test_x1, test_x2 = np.meshgrid(test1d, test1d)
test = np.hstack([np.vstack(test_x1.ravel()), np.vstack(test_x2.ravel())])

print(np.shape(train))
print(np.shape(test))
print(np.shape(target))

# Store standard deviations of the training data and targets.
stdx = np.std(train)
stdy = np.std(target)
sdt1 = np.sqrt(0.1)

# Standardize the training and test data on the same scale.
std = standardize(train_matrix=train,
                  test_matrix=test)

# Plotting.
plt3d = plt.figure().gca(projection='3d')

# Plot training data.
plt3d.scatter(train[:, 0], train[:, 1], target,  color='b')

# Plot exact function.
plt3d.plot_surface(test_x1, test_x2,
                   afunc(test).reshape(np.shape(test_x1)),
                   alpha=0.3, color='b')

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
    for tf, tt in zip(std['test'], afunc(test)):
        p = ((np.dot(coef, tf)) * target_std['std']) + target_std['mean']
        rr_predictions.append(p)
        sumd += (p - tt) ** 2
    print('Ridge regression prediction:', (sumd / len(test)) ** 0.5)
    # Plot the prediction.
    plt3d.plot_surface(test_x1, test_x2,
                       np.array(rr_predictions).reshape(np.shape(test_x1)),
                       alpha=0.3, color='r')

if True:
    # Model example 2 - Gausian linear kernel regression.
    # Define prediction parameters
    kdict = {'k1': {'type': 'linear', 'scaling': 0.9},
             'c1': {'type': 'constant', 'const': 0.0}}
    # Set up the prediction routine.
    gp1 = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                          train_fp=std['train'], train_target=target,
                          optimize_hyperparameters=True,
                          scale_optimizer=False)
    # Do predictions.
    linear = gp1.predict(test_fp=std['test'], get_validation_error=True,
                         test_target=afunc(test))
    print('Gaussian linear regression prediction:',
          linear['validation_error']['absolute_average'])
    # Plot the prediction.
    plt3d.plot_surface(test_x1, test_x2,
                       linear['prediction'].reshape(np.shape(test_x1)),
                       alpha=0.3, color='g')


if True:
    # Model example 3 - Gaussian Process with sqe kernel.
    # Set up the prediction routine and optimize hyperparameters.
    kdict = {'k1': {'type': 'gaussian', 'width': [0.3, 3]},
             'c1': {'type': 'constant', 'const': 0.0}}
    gp2 = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                          train_fp=std['train'], train_target=target,
                          optimize_hyperparameters=True,
                          scale_optimizer=False)
    # Do the optimized predictions.
    optimized = gp2.predict(test_fp=std['test'], get_validation_error=True,
                            test_target=afunc(test))
    print('Gaussian process prediction:',
          optimized['validation_error']['absolute_average'])
    # Plot the prediction.
    plt3d.plot_surface(test_x1, test_x2,
                       optimized['prediction'].reshape(np.shape(test_x1)),
                       alpha=0.3, color='g')

plt.xlabel('Descriptor 0')
plt.ylabel('Descriptor 1')
plt.axis('tight')
plt.show()
