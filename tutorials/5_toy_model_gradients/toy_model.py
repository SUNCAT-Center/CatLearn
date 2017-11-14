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
from atoml.preprocess.scale_target import target_standardize
from atoml.regression import GaussianProcess
from atoml.utilities.cost_function import get_error

# First derivative observations can be included.
eval_gradients = True

# A known underlying function in one dimension and it's first derivative.
def afunc(x):
    """ Function [y] and first derivative [dy] """
    y =  np.sin(x)*np.cos(x)*np.exp(2*x)*x**-2*np.exp(-x)*np.cos(x)*np.sin(x)
    dy =  (2 * np.exp(x)*np.cos(x)**3* np.sin(x))/x**2-(2*np.exp(x)* np.cos(
    x)**2* np.sin(x)**2)/x**3+(np.exp(x)*np.cos(x)**2 *np.sin(x)**2)/x**2-(
    2*np.exp(x)*np.cos(x)*np.sin(x)**3)/x**2
    return [y,dy]

# Setting up data.

# A number of training points in x.
train = np.array([[1.0, 2.0, 4.0, 4.8, 5.0, 5.5, 5.8, 6.0]])


# Each element in the list train can be referred to as a fingerprint.

# Call the underlying function to produce the target values.
target = np.array(afunc(train)[0])

# Generate test datapoints x.
test_points = 100
test = np.linspace(0.1,6.0,test_points)

# Store standard deviations of the training data and targets.
# stdx = np.std(train)
# stdy = np.std(target)
# tstd = 2.


# Train and test data without standardizing.

train = np.reshape(train, (np.shape(train)[1], 1))
test =np.reshape(test, (test_points, 1))

# # Standardize the training and test data on the same scale.
# std = standardize(train_matrix=np.reshape(train, (np.shape(train)[1], 1)),
#                   test_matrix=np.reshape(test, (test_points, 1)))
# # Standardize the training targets.
# train_targets = target_standardize(target[0])

# Note that predictions will now be made on the standardized scale.

org_target = target.copy()
org_train = train.copy()
org_test = test.copy()

# Gradients

if eval_gradients == True:
    gradients = []
    for i in train:
        gradients.append(afunc(i)[1])
    org_gradients = np.asarray(gradients)
    y_tilde = []
    # Build y_tilde as:
    # y_tilde = y1, y2...yN, delta1, delta2...deltaN
    y_tilde = np.append(target, gradients)
    y_tilde = np.reshape(y_tilde,(np.shape(y_tilde)[0],1))
    target = y_tilde
    target = np.reshape(target,(np.shape(target)[1],np.shape(target)[0]))


# Model example - Gaussian Process

# Define prediction parameters.
sdt1 = 0.01
w1 = 1.0  # Too large widths results in a biased model.

# Set up the prediction routine and optimize hyperparameters.
kdict = {'k1': {'type': 'gaussian', 'width': [w1], 'scaling': 1.0}}
gp = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                     train_fp=train,
                     train_target=target[0],
                     optimize_hyperparameters=True,
                     eval_gradients=eval_gradients)
print('Optimized kernel:', gp.kernel_dict)
# Do the optimized predictions.
optimized = gp.predict(test_fp=test, uncertainty=True)
# Scale predictions back to the original scale.

# opt_prediction = np.array(optimized['prediction']) * train_targets['std'] + \
#     train_targets['mean']
# opt_uncertainty = np.array(optimized['uncertainty']) * train_targets['std']
opt_prediction = np.array(optimized['prediction'])
opt_uncertainty = np.array(optimized['uncertainty'])

# Get average errors.
error = get_error(opt_prediction, afunc(test)[0])
print('Gaussian linear regression prediction:', error['absolute_average'])
# Get confidence interval on predictions.
opt_upper = opt_prediction + opt_uncertainty #* tstd
opt_lower = opt_prediction - opt_uncertainty #* tstd

# Plotting.
# Store the known underlying function for plotting.

linex = np.linspace(0.1,6.0,1000)
linex = np.reshape(linex, (1,np.shape(linex)[0]))
linex = np.sort(linex)
liney = []
for i in linex:
    liney.append(afunc(i)[0])

fig = plt.figure(figsize=(5, 5))

# Example
ax = fig.add_subplot(111)
ax.plot(linex[0], liney[0], '-', lw=1, color='black')
ax.plot(train, org_target[0], 'o', alpha=0.2, color='black')
ax.plot(test, opt_prediction, 'g-', lw=1, alpha=0.4)
ax.fill_between(test[:,0], opt_upper, opt_lower, interpolate=True, \
color='green',
                alpha=0.2)
plt.title('Optimized GP. \n w: {0:.3f}, r: {1:.3f}'.format(
    gp.kernel_dict['k1']['width'][0], np.sqrt(gp.regularization)))
plt.xlabel('Descriptor')
plt.ylabel('Response')
plt.axis('tight')

# Gradients

if eval_gradients==True:
    org_target = np.reshape(org_target,(np.shape(org_target)[1], np.shape(
    org_target)[0]))
    size_bar_gradients = (np.abs(np.max(linex) - np.min(linex))/2.0)/25.0
    def lineary(m,linearx,train,target):
            """Define some linear function."""
            lineary = m*(linearx-train)+target
            return lineary
    for i in range(0,np.shape(org_gradients)[0]):
        linearx_i = np.linspace(org_train[i]-size_bar_gradients, org_train[i]+
        size_bar_gradients,num=10)
        lineary_i = lineary(org_gradients[i],linearx_i,org_train[i],
        org_target[i])
        ax.plot(linearx_i, lineary_i, '-', lw=3, alpha=0.5, color='black')


plt.show()
