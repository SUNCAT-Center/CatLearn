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
    return 3. * x[:, 0] - x[:, 1]

# Setting up data.
# A number of training points in x.
train_points = 30

# Randomly generate the training datapoints x.
train_d1 = 2 * np.random.random_sample(train_points) - 0.5
train_d2 = 2 * np.random.random_sample(train_points) - 0.5
train_x1, train_x2 = np.meshgrid(train_d1, train_d2)
train = np.hstack([np.vstack(train_d1), np.vstack(train_d2)])

# Each element in the list train can be referred to as a fingerprint.
# Call the underlying function to produce the target values.
target = afunc(train)

# Add random noise from a normal distribution to the target values.
nt = []
for i in range(train_points):
    nt.append(0.1*np.random.normal())
target += np.array(nt)
target = np.reshape(target, (len(target), 1))

# Generate test datapoints x.
test_points = 101
test1d = np.vstack(np.linspace(-1.3, 1.3, test_points))
test_x1, test_x2 = np.meshgrid(test1d, test1d)
test = np.hstack([np.vstack(test1d), np.vstack(test1d)])

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
if False:
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
        p = (np.dot(coef, tf))
        rr_predictions.append(p)
        sumd += (p - tt) ** 2
    print('Ridge regression prediction:', (sumd / len(test)) ** 0.5)


# Model example 2 - Gausian linear kernel regression.
# Define prediction parameters
sdt1 = np.sqrt(1e-5)
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

# Model example 3 - Gaussian Process with sqe kernel.
# Set up the prediction routine and optimize hyperparameters.
kdict = {'k1': {'type': 'gaussian', 'width': 0.5}}
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
linex1 = np.linspace(np.min(train), np.max(train), test_points)
liney = afunc(np.hstack([np.vstack(linex1), np.vstack(linex1)]))

plt3d = plt.figure().gca(projection='3d')
plt3d.scatter(train[:, 0], train[:, 1], target,  color='green')

# Example 1 - Ridge regression.
plt3d.plot_surface(test_x1, test_x2, afunc(test), alpha=0.2, color='r')
plt3d.plot_surface(test_x1, test_x2, linear['prediction'], alpha=0.2, color='b')
plt.xlabel('Descriptor 0')
plt.ylabel('Descriptor 1')
#plt.zlabel('Response')
plt.axis('tight')
plt.show()

# Example 2
fig = plt.figure(figsize=(15, 8))
#ax = fig.add_subplot(223)
#ax.plot(linex, liney, '-', lw=1, color='black')
#ax.plot(train[0], target[0], 'o', alpha=0.2, color='black')
#ax.plot(test, optimized['prediction'], 'g-', lw=1, alpha=0.4)
#ax.fill_between(test, opt_upper, opt_lower, interpolate=True, color='green',
#                alpha=0.2)
#plt.title('Optimized GP. \n w: {0:.3f}, r: {1:.3f}'.format(
#    gp2.kernel_dict['k1']['width'][0]*stdx, np.sqrt(gp2.regularization)*stdy))
#plt.xlabel('Descriptor')
#plt.ylabel('Response')
#plt.axis('tight')

# Uncertainty profile.
#ax = fig.add_subplot(224)
#ax.plot(test, np.array(optimized['uncertainty'] * tstd), '-', lw=1,
#        color='green')
#plt.title('Uncertainty Profiles')
#plt.xlabel('Descriptor')
#plt.ylabel('Uncertainty')
#plt.axis('tight')
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                    wspace=0.4, hspace=0.4)
#plt.show()
