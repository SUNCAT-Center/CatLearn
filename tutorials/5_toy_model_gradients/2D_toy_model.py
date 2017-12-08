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


# Set whether first derivative observations are included or not.

# A known underlying 2D function (z) and first derivatives (dx,dy).
def afunc(x,y):
    """S FUNCTION"""
    z = -(12.0)*(x**2.0) + (1.0/3.0)*(x**4.0)
    z = z -(12.0)*(y**2.0) + (1.0/2.0)*(y**4.0)
    dx = -24.0*x + (4.0/3.0)*x**3.0
    dy = -24.0*y + 2.0*y**3.0
    return [z,dx,dy]

# Setting up data.

# A number of training points in x and y.
train_points = 5 # Number of training points.

# Each element in the matrix train can be referred to as a fingerprint.
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

# Generate test datapoints in x and y.
test_points = 50 # Length of the grid test points (nxn).

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

# Call the underlying function to produce the gradients of the target values.
gradients = []
for i in org_train:
    gradients.append(afunc(i[0],i[1])[1:3])
org_gradients = np.asarray(gradients)
gradients = org_gradients

# Gaussian Process.

# Define prediction parameters.
sdt1 = 0.01
w1 = 1.0  # Too large widths results in a biased model.
scaling = 1.0

kdict = {'k1': {'type': 'gaussian', 'width': w1, 'scaling': scaling}}

gp = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                     train_fp=train,
                     train_target=target,
                     optimize_hyperparameters=False,
                     gradients=gradients,
                     scale_optimizer=False,scale_data=True)
print('Optimized kernel:', gp.kernel_dict)

# Hyperaparam optimization algorithms change from default.
gp.optimize_hyperparameters(algomin='TNC',global_opt=False)

# Do the optimized predictions.
pred = gp.predict(test_fp=test, uncertainty=True)
prediction = np.array(pred['prediction'][:,0])

# Calculate the uncertainty of the predictions.
uncertainty = np.array(pred['uncertainty'])

# Get confidence interval on predictions.
upper = prediction + uncertainty
lower = prediction - uncertainty

interval = upper - prediction

# Plots.

plt.figure(figsize=(11.0, 5.0))

# Contour plot for real function.
plt.subplot(131)

x = np.linspace(-5.0, 5.0, test_points)
y = np.linspace(-5.0, 5.0, test_points)
X,Y = np.meshgrid(x, y)
plt.contourf(X, Y, afunc(X, Y)[0]-np.min(afunc(X, Y)[0]), 6, alpha=.70, cmap='PRGn')
cbar = plt.colorbar(orientation="horizontal", pad=0.15)
cbar.set_label('Response', rotation=0)
C = plt.contour(X, Y, afunc(X, Y)[0]-np.min(afunc(X, Y)[0]), 6, colors='black', linewidths=1)
plt.clabel(C, inline=1, fontsize=9)
plt.title('Real function',fontsize=10)
plt.xlabel('Descriptor 1')
plt.ylabel('Descriptor 2')


# Contour plot for predicted function.
plt.subplot(132)

x = []
for i in range(len(test)):
    t = org_test[i][0]
    t = x.append(t)
y = []
for i in range(len(test)):
    t = org_test[i][1]
    t = y.append(t)

zi = plt.mlab.griddata(x, y, prediction-np.min(prediction), testx, testy,
interp='linear')
plt.contourf(testx, testy, zi, 6, alpha=.70, cmap='PRGn')
cbar = plt.colorbar(orientation="horizontal", pad=0.15)
cbar.set_label('Response', rotation=0)
C = plt.contour(testx, testy, zi, 6, colors='k',linewidths=1.0)
plt.clabel(C, inline=0.1, fontsize=9)

# Print training points positions (circles)
plt.scatter(org_train[:,0],org_train[:,1],marker='o',s=5.0,c='black',
edgecolors='black',alpha=0.8)
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.title('GP predicted function',fontsize=10)
plt.xlabel('Descriptor 1')


# Plot uncertainty
plt.subplot(133)
zi = plt.mlab.griddata(x, y, interval, testx, testy,
interp='linear')
plt.title('GP uncertainty',fontsize=10)
plt.contourf(testx, testy, zi, 6, alpha=.70, cmap='Reds',vmin=0,
vmax=np.max(afunc(X,Y)[0]))
cbar = plt.colorbar(orientation="horizontal", pad=0.15)
cbar.set_label('Uncertainty', rotation=0)
C = plt.contour(testx, testy, zi, 6, colors='k',linewidths=1.0)
plt.clabel(C, inline=0.1, fontsize=9)
plt.xlabel('Descriptor 1')




plt.show()