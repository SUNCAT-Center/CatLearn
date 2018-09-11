"""This tutorial is intended to show that the resulting estimates are
improved by including first derivative observations.

First we set up a known underlying function in one dimension.
Then, we pick some values to train.
Finally we will use CatLearn to make predictions on some unseen fingerprint and
benchmark those predictions against the known underlying function.
"""
import numpy as np
import matplotlib.pyplot as plt
from catlearn.regression import GaussianProcess
from catlearn.regression.cost_function import get_error


# A known underlying function in one dimension (y) and first derivative (dy).
def afunc(x):
    """Function (y) and first derivative (dy)."""
    y = 10000 * (10 + (x-4) * np.sin(x))
    for i in y:
        i += np.random.normal(scale=0.25)
    dy = 10000 *((x-4) * np.cos(x) + np.sin(x))
    return [y, dy]


# Setting up data.

# A number of training points in x.
train_points = 4

# Each element in the list train can be referred to as a fingerprint.

train = [[1.0]]
train = np.concatenate((train, [[0.0], [2.0], [4.0], [6.0], [7.0]]))

# Call the underlying function to produce the target values.
target = np.array(afunc(train)[0])

# Generate test datapoints x.
test_points = 500
test = np.linspace(0.0, 7.0, test_points)
test = np.reshape(test, (test_points, 1))

# Make a copy of the original features and targets.
org_train = train.copy()
org_target = target.copy()
org_test = test.copy()

# Call the underlying function to produce the gradients of the target values.

gradients = []
for i in org_train:
    gradients.append(afunc(i)[1])
org_gradients = np.asarray(gradients)
gradients = org_gradients

# Gaussian Process.

# Define initial prediction parameters.

mean_target = np.mean(target)
target = target.copy() - mean_target

sigma_n = 0.2 # Regularisation parameter.
w1 = 1.0  # Length scale parameter.
scaling = np.std(target)**2 + 20000.  # Scaling parameter.
print('standard:', scaling)
# Prior
sigma_c = 0.0

# Set up the prediction routine and optimize hyperparameters.
# Note that the default algorithm is L-BFGS-B but one could use also TNC.
# Note also that global optimisation using basinhopping can be activated when
# setting the flag global_opt=True.

kdict = {'k1': {'type': 'gaussian', 'width': w1, 'dimension':'features',
                'bounds': ((w1, w1),),
                'scaling': scaling,
                'scaling_bounds': ((scaling, scaling+10.0),),
                },
         'k2': {'type': 'constant',
                    'const': sigma_c,
                    'bounds': ((sigma_c, sigma_c),)}
         }

gp = GaussianProcess(
    kernel_dict=kdict, regularization=sigma_n, regularization_bounds=(sigma_n,
    sigma_n),
    train_fp=train,
    train_target=target, gradients=gradients, optimize_hyperparameters=False,
    scale_data=False)

gp.optimize_hyperparameters(global_opt=False)
print('Optimized hyperparameters:', gp.kernel_dict)
print('logmarg', gp.log_marginal_likelihood)
# Do the optimized predictions.
pred = gp.predict(test_fp=test, uncertainty=True)
prediction = np.array(pred['prediction'][:, 0]) + mean_target

# Calculate the uncertainty of the predictions.
uncertainty = np.sqrt(np.array(pred['uncertainty']))

# Get confidence interval on predictions.
upper = prediction + uncertainty
lower = prediction - uncertainty

# Get average errors.
error = get_error(prediction, afunc(test)[0])
print('Gaussian linear regression prediction:', error['absolute_average'])

# Plotting.
# Store the known underlying function for plotting.

linex = np.linspace(0.0, 7.0, test_points)
linex = np.reshape(linex, (1, np.shape(linex)[0]))
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
ax.fill_between(org_test[:, 0], upper, lower, interpolate=True, color='red',
                alpha=0.2)
plt.title('GP. \n w: {0:.3f}, r: {1:.3f}'.format(
    gp.kernel_dict['k1']['width'][0], np.sqrt(gp.regularization)))
plt.xlabel('Descriptor')
plt.ylabel('Response')
plt.axis('tight')

# Plot gradients (when included).

if gradients is not None:
    size_bar_gradients = (np.abs(np.max(linex) - np.min(linex))/2.0)/25.0

    def lineary(m, linearx, train, target):
            """Define some linear function."""
            lineary = m*(linearx-train)+target
            return lineary

    for i in range(0, np.shape(org_gradients)[0]):
        linearx_i = np.linspace(
            org_train[i]-size_bar_gradients,
            org_train[i]+size_bar_gradients, num=10)
        lineary_i = lineary(org_gradients[i], linearx_i, org_train[i],
                            org_target[i])
        ax.plot(linearx_i, lineary_i, '-', lw=3, alpha=0.5, color='black')


plt.show()
