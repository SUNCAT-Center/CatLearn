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
eval_gradients = False
train_points = 6 # Number of training points.
test_points = 25 # Length of the grid test points (nxn).

# A known underlying function in one dimension [y] and first derivative [dy].
def afunc(x,y):
    """S FUNCTION"""
    z = -(12.0)*(x**2.0) + (1.0/3.0)*(x**4.0)
    z = z -(12.0)*(y**2.0) + (1.0/2.0)*(y**4.0)
    dx = -24.0*x + (4.0/3.0)*x**3.0
    dy = -24.0*y + 2.0*y**3.0
    return [z,dx,dy]

# Setting up data.

# A number of training points in x.
# Each element in the list train can be referred to as a fingerprint.

train = []
trainx = np.linspace(-5.0, 5.0, train_points)
trainy = np.linspace(-5.0, 5.0, train_points)
for i in range(len(trainx)):
    for j in range(len(trainy)):
        train1 = [trainx[i],trainy[j]]
        train.append(train1)
train = np.array(train)
train = np.append(train,([[0.0,0.0]]),axis=0)


# Call the underlying function to produce the target values.
target = []
for i in train:
    target.append([afunc(i[0],i[1])[0]])
target = np.array(target)


# Generate test datapoints x.
test = []
testx = np.linspace(-5.0, 5.0, test_points)
testy = np.linspace(-5.0, 5.0, test_points)
for i in range(len(testx)):
    for j in range(len(testy)):
        test1 = [testx[i],testy[j]]
        test.append(test1)
test = np.array(test)


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

gp = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                     train_fp=train,
                     train_target=target,
                     optimize_hyperparameters=True,
                     eval_gradients=eval_gradients, algomin='TNC',
                     global_opt=True)
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


# Plots.


plt.figure(figsize=(8.0, 5.0))

# Contour plot for real function.

plt.subplot(121)
x = np.linspace(-5.0, 5.0, test_points)
y = np.linspace(-5.0, 5.0, test_points)
X,Y = np.meshgrid(x, y)
plt.contourf(X, Y, afunc(X, Y)[0], 6, alpha=.70, cmap='PRGn',vmin=np.min(afunc(
X,Y)[0]),
vmax=np.max(afunc(X,Y)[0]))
plt.colorbar(orientation="horizontal", pad=0.1)
plt.clim(np.min(afunc(
X,Y)[0]),np.max(afunc(
X,Y)[0]))
C = plt.contour(X, Y, afunc(X, Y)[0], 6, colors='black', linewidths=1)
plt.clabel(C, inline=1, fontsize=9)
plt.title('Real function',fontsize=10)

# Contour plot for predicted function.

plt.subplot(122)

x = []
for i in range(len(test)):
    t = org_test[i][0]
    t = x.append(t)
y = []
for i in range(len(test)):
    t = org_test[i][1]
    t = y.append(t)


zi = plt.mlab.griddata(x, y, prediction, testx, testy, interp='linear')

plt.contourf(testx, testy, zi, 6, alpha=.70, cmap='PRGn',vmin=np.min(afunc(
X,Y)[0]),
vmax=np.max(afunc(X,Y)[0]))
plt.colorbar(orientation="horizontal", pad=0.1)
C = plt.contour(testx, testy, zi, 6, colors='k',linewidths=1.0)
plt.clabel(C, inline=0.1, fontsize=9)


plt.show()