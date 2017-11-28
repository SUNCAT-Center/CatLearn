""" This tutorial is intended to help you get familiar with using AtoML to set
up a model and do predictions.

First we set up a known underlying function in one dimension (including
first derivative). Then, we pick some values to train.
Finally we will use AtoML to make predictions on some unseen fingerprint and
benchmark those predictions against the known underlying function.
"""
import numpy as np
import matplotlib.pyplot as plt
from atoml.preprocess.feature_preprocess import standardize
from atoml.preprocess.scale_target import target_standardize
from atoml.regression import GaussianProcess
from atoml.utilities.cost_function import get_error


# The user can choose whether the features and/or the targets are standardized.
StandardizeFeatures = True
StandardizeTargets = True

# First derivative observations can be included.
eval_gradients = True


# A known underlying function in one dimension [y] and first derivative [dy].
def afunc(x):
    """ Function [y] and first derivative [dy] """
    y =  np.sin(x)*np.cos(x)*np.exp(2*x)*x**-2*np.exp(-x)*np.cos(x)*np.sin(x)
    dy =  (2 * np.exp(x)*np.cos(x)**3* np.sin(x))/x**2-(2*np.exp(x)* np.cos(
    x)**2* np.sin(x)**2)/x**3+(np.exp(x)*np.cos(x)**2 *np.sin(x)**2)/x**2-(
    2*np.exp(x)*np.cos(x)*np.sin(x)**3)/x**2
    return [y,dy]

# Setting up data.

# A number of training points in x.
# Each element in the list train can be referred to as a fingerprint.
train = np.array([[1.0], [2.0], [3.0], [4.0]])

# Call the underlying function to produce the target values.
target = np.array(afunc(train)[0])

# Generate test datapoints x.
test_points = 100
test = np.linspace(0.1,6.0,test_points)
test = np.reshape(test, (test_points, 1))

# Make a copy of the original features and targets.

org_train = train.copy()
org_target = target.copy()
org_test = test.copy()

# Standardization of the train, test and target data.

if StandardizeFeatures:
    feature_std = standardize(train_matrix=train,test_matrix=test)
    train, train_mean, train_std = feature_std['train'], feature_std['mean'], \
    feature_std['std']
    test = (test -train_mean)/train_std

else:
    train_mean, train_std = 0.0, 1.0
if StandardizeTargets:
    target_std = target_standardize(target)
    target, target_mean, target_std = target_std['target'], target_std[
    'mean'], target_std['std']
else:
    target_mean, target_std = 0.0, 1.0

# Call the underlying function to produce the gradients of the target values.

if eval_gradients:
    gradients = []
    for i in org_train:
        gradients.append(afunc(i)[1])
    org_gradients = np.asarray(gradients)
    gradients = org_gradients/(target_std/train_std)
    y_tilde = np.append(target, gradients)
    target = np.reshape(y_tilde,(np.shape(y_tilde)[0],1))

# Gaussian Process.

# Define prediction parameters.
sdt1 = 0.01
w1 = 1.0  # Too large widths results in a biased model.

# Set up the prediction routine and optimize hyperparameters.
kdict = {'k1': {'type': 'gaussian', 'width': [w1], 'scaling': 1.0}}

# kdict = {'k1': {'type': 'gaussian', 'width': [w1], 'scaling': 1.0}, 'k2': {
# 'type': 'linear', 'scaling': 1.0}}

# kdict = {'k1': {'type': 'gaussian', 'width': [w1], 'scaling': 1.0}, 'k2': {
# 'type': 'constant','const': 1.0, 'scaling': 1.0}}

gp = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                     train_fp=train,
                     train_target=target,
                     optimize_hyperparameters=True,
                     eval_gradients=eval_gradients, algomin='TNC',
                     global_opt=False)
print('Optimized kernel:', gp.kernel_dict)

# Do the optimized predictions.
pred = gp.predict(test_fp=test, uncertainty=True)
prediction = np.array(pred['prediction'][:,0])

# Calculate the uncertainty of the predictions.
uncertainty = np.array(pred['uncertainty'])

# Get confidence interval on predictions.
upper = prediction + uncertainty
lower = prediction - uncertainty

# Scale predictions back to the original scale.
if StandardizeTargets:
    uncertainty = (uncertainty*target_std) + target_mean
    prediction = (prediction*target_std) + target_mean
    upper = (upper*target_std) + target_mean
    lower = (lower*target_std) + target_mean

# Get average errors.
error = get_error(prediction, afunc(test)[0])
print('Gaussian linear regression prediction:', error['absolute_average'])

# Plotting.

# Store the known underlying function for plotting.

linex = np.linspace(0.1,6.0,test_points)
linex = np.reshape(linex, (1,np.shape(linex)[0]))
linex = np.sort(linex)
liney = []
for i in linex:
    liney.append(afunc(i)[0])

fig = plt.figure(figsize=(5, 5))

# Example
ax = fig.add_subplot(111)
ax.plot(linex[0], liney[0], '-', lw=1, color='black')
ax.plot(org_train, org_target, 'o', alpha=0.2, color='black')
ax.plot(org_test, prediction, 'g-', lw=1, alpha=0.4)
ax.fill_between(org_test[:,0], upper, lower, interpolate=True,
color='green', alpha=0.2)
# plt.title('GP. \n w: {0:.3f}, r: {1:.3f}'.format(
#     gp.kernel_dict['k1']['width'][0], np.sqrt(gp.regularization)))
plt.xlabel('Descriptor')
plt.ylabel('Response')
plt.axis('tight')

# Gradients

if eval_gradients==True:
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
