#!/usr/bin/env python
# coding: utf-8

# # Fourth CatLearn tutorial.
# 
# This tutorial is intended to give further intuition for Gaussian processes.
# 
# Like in tutorial 3, we set up a known underlying function with two training features and one output feature, we generate training and test data and calculate predictions and errors.
# 
# We will compare the results of linear ridge regression, Gaussian linear kernel regression and finally a Gaussian process with the usual squared exponential kernel.

# In[1]:


# Import packages.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from catlearn.preprocess.scaling import standardize, target_standardize
from catlearn.regression import GaussianProcess
from catlearn.regression.cost_function import get_error


# In[2]:


# A known underlying function in two dimensions
def afunc2d(x):
    """2D linear function (plane)."""
    return 3. * x[:, 0] - 1. * x[:, 1] + 50.


# In[3]:


# Setting up data.
# A number of training points in x.
train_points = 30
# Magnitude of the noise.
noise_magnitude = 0.3

# Randomly generate the training datapoints x.
train_d1 = 2 * (np.random.random_sample(train_points) - 0.5)
train_d2 = 2 * (np.random.random_sample(train_points) - 0.5)
train_x1, train_x2 = np.meshgrid(train_d1, train_d2)
train = np.hstack([np.vstack(train_d1), np.vstack(train_d2)])

# Each element in the list train can be referred to as a fingerprint.
# Call the underlying function to produce the target values.
target = np.array(afunc2d(train))

# Add random noise from a normal distribution to the target values.
for i in range(train_points):
    target[i] += noise_magnitude * np.random.normal()

# Generate test datapoints x.
test_points = 30
test1d = np.vstack(np.linspace(-1.5, 1.5, test_points))
test_x1, test_x2 = np.meshgrid(test1d, test1d)
test = np.hstack([np.vstack(test_x1.ravel()), np.vstack(test_x2.ravel())])

print(np.shape(train))
print(np.shape(test))
print(np.shape(target))

# Standardize the training and test data on the same scale.
std = standardize(train_matrix=train,
                  test_matrix=test)
# Standardize the training targets.
train_targets = target_standardize(target)
# Note that predictions will now be made on the standardized scale.

# plot predicted tstd * uncertainties intervals.
tstd = 2.


# The `tstd` variable specifies how many standard deviations we plot.
# 
# # Model example 1 - Gaussian linear kernel regression.

# In[4]:


# Define prediction parameters
kdict = [{'type': 'linear', 'scaling': 1.},
         {'type': 'constant', 'const': 0.}]
# Starting guess for the noise parameter
sdt1 = noise_magnitude
# Set up the gaussian process.
gp1 = GaussianProcess(kernel_list=kdict, regularization=np.sqrt(sdt1),
                      train_fp=std['train'],
                      train_target=train_targets['target'],
                      optimize_hyperparameters=True)
# Do predictions.
linear = gp1.predict(test_fp=std['test'], uncertainty=True)
# Put predictions back on real scale.
prediction = np.vstack(linear['prediction']) * train_targets['std'] +     train_targets['mean']
# Put uncertainties back on real scale.
uncertainty = np.vstack(linear['uncertainty_with_reg']) * train_targets['std']
# Get confidence interval on predictions.
over_upper = prediction + uncertainty * tstd
over_lower = prediction - uncertainty * tstd
# Plotting.
plt3d = plt.figure(0).gca(projection='3d')

# Plot training data.
plt3d.scatter(train[:, 0], train[:, 1], target,  color='b')

# Plot exact function.
plt3d.plot_surface(test_x1, test_x2,
                   afunc2d(test).reshape(np.shape(test_x1)),
                   alpha=0.3, color='b')
# Plot the uncertainties upper and lower bounds.
plt3d.plot_surface(test_x1, test_x2,
                   over_upper.reshape(np.shape(test_x1)),
                   alpha=0.3, color='r')
plt3d.plot_surface(test_x1, test_x2,
                   over_lower.reshape(np.shape(test_x1)),
                   alpha=0.3, color='r')


# We see the upper and lower bounds of the confidence interval predicted by the linear model. It is fairly confident on even beyond the region of the training data set.
# 
# # Model example 2 - squared exponential kernel.

# In[5]:


# Set up the prediction routine and optimize hyperparameters.
kdict = [{'type': 'gaussian', 'width': [0.3, 3.]}]
# Starting guess for the noise parameter
sdt1 = noise_magnitude
# Set up the gaussian process.
gp2 = GaussianProcess(kernel_list=kdict, regularization=np.sqrt(sdt1),
                      train_fp=std['train'],
                      train_target=train_targets['target'],
                      optimize_hyperparameters=True)
# Do the optimized predictions.
gaussian = gp2.predict(test_fp=std['test'], uncertainty=True)
# Put predictions back on real scale.
prediction = np.vstack(gaussian['prediction']) * train_targets['std'] +     train_targets['mean']
# Put uncertainties back on real scale.
uncertainty = np.vstack(gaussian['uncertainty_with_reg']) * train_targets['std']
# Get confidence interval on predictions.
gp_upper = prediction + uncertainty * tstd
gp_lower = prediction - uncertainty * tstd
# Plotting.
plt3d = plt.figure(1).gca(projection='3d')

# Plot training data.
plt3d.scatter(train[:, 0], train[:, 1], target,  color='b')

# Plot exact function.
plt3d.plot_surface(test_x1, test_x2,
                   afunc2d(test).reshape(np.shape(test_x1)),
                   alpha=0.3, color='b')
# Plot the prediction.
plt3d.plot_surface(test_x1, test_x2,
                   gp_upper.reshape(np.shape(test_x1)),
                   alpha=0.3, color='g')
plt3d.plot_surface(test_x1, test_x2,
                   gp_lower.reshape(np.shape(test_x1)),
                   alpha=0.3, color='g')


# Here we can see the confidence interval grows much faster, revealing that the squared exponential kernel is more uncertain outside the region of the training data.
# 
# ### Experiment and get intuition.
# 
# Now, try playing around with the `train_points`, `noise_magnitude`  and `tstd` variables and rerun the models, to get a feel for the behavior of the Gaussian process and for viewing the various levels of confidence the two models can achieve. 
# 
# 
# # Types of uncertainty
# 
# A gaussian process gives estimates epistemic uncertainty and also a constant random noise associated with the observations is assumed, giving us homoscedastic uncertainty. Heteroscedastic uncertainty is not constant although it may be random. It could have an underlying trend making observations less certain in some regions of space than others. Let us try adding heteroscedastic uncertainty to a toy model.

# In[6]:


# A known underlying function in one dimension.
def afunc(x):
    """Define some polynomial function."""
    y = x - 50.
    p = (y + 4) * (y + 4) * (y + 1) * (y - 1) * (y - 3.5) * (y - 2) * (y - 1)
    p += 40. * y + 80. * np.sin(10. * x)
    return 1. / 20. * p + 500


# In[7]:


# A number of training points in x.
train_points = 513
noise_magnitude = 1.
a_noise_magnitude = 10.
a_noise_center = 49.
a_noise_width = 1.

# Randomly generate the training datapoints x.
train = 7.6 * np.random.sample((train_points, 1)) - 4.2 + 50

# Each element in the list train can be referred to as a fingerprint.
# Call the underlying function to produce the target values.
target = np.array(afunc(train))

# Add random noise from a normal distribution to the target values.
target += noise_magnitude * np.random.randn(train_points, 1)
# Heteroscedastic uncertainty:
def au(x):
    out = a_noise_magnitude * np.exp(-(x-a_noise_center)**2 / (2 * a_noise_width ** 2))
    return out
target += np.random.randn(train_points, 1) * au(train)

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


# In[8]:


# Set up the prediction routine and optimize hyperparameters.
w3 = 0.1
sdt3 = 0.01
kdict = [{'type': 'gaussian', 'width': [w3]}]
gp = GaussianProcess(kernel_list=kdict, regularization=sdt3,
                     train_fp=std['train'],
                     train_target=train_targets['target'],
                     optimize_hyperparameters=True)
print('Optimized kernel:', gp.kernel_list)
print(-gp.theta_opt['fun'])
# Do the optimized predictions.
optimized = gp.predict(test_fp=std['test'], uncertainty=True)

# Scale predictions back to the original scale.
opt_prediction = np.vstack(optimized['prediction']) *     train_targets['std'] + train_targets['mean']
opt_uncertainty = np.vstack(optimized['uncertainty_with_reg']) *     train_targets['std']

# Get average errors.
error = get_error(opt_prediction.reshape(-1), afunc(test).reshape(-1))
print('Gaussian kernel regression prediction:', error['absolute_average'])

# Get confidence interval on predictions.
opt_upper = opt_prediction + opt_uncertainty * tstd
opt_lower = opt_prediction - opt_uncertainty * tstd

# Plot eample 3
plt.figure(2)
plt.plot(linex, liney, '-', lw=1, color='black')
plt.plot(train, target, 'o', alpha=0.2, color='black')
plt.plot(test, opt_prediction, 'g-', lw=1, alpha=0.4)
plt.fill_between(np.hstack(test), np.hstack(opt_upper),
                np.hstack(opt_lower), interpolate=True,
                color='green', alpha=0.2)
plt.title('Optimized GP. \n w: {0:.3f}, r: {1:.3f}'.format(
    gp.kernel_list[0]['width'][0] * stdx,
    np.sqrt(gp.regularization) * stdy))
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('tight')


# Observe how the area around `a_noise_center` is more noisy.
# 
# Our GP uncertainty is biased, when the data is not equally noisy everywhere. How do we heteroscedastic the epistemic uncertainty?
# There are several ways. One that does not require any further changes to the GP code is fitting another GP to the magnitude of error.

# In[9]:


# Get training errors.
optimized_train = gp.predict(test_fp=std['train'], uncertainty=True)
predict_train = np.vstack(optimized_train['prediction']) *     train_targets['std'] + train_targets['mean']
uncertainty_train = np.vstack(optimized_train['uncertainty_with_reg']) *     train_targets['std']
train_error = get_error(predict_train.reshape(-1), afunc(train).reshape(-1))
gp_heteroscedastic = GaussianProcess(kernel_list=kdict,
                                     regularization=sdt3,
                                     train_fp=std['train'],
                                     train_target=train_error['rmse_all']-uncertainty_train.reshape(-1),
                                     optimize_hyperparameters=True)
gp.optimize_hyperparameters(global_opt=True)
heteroscedastic_uncertainty = gp_heteroscedastic.predict(test_fp=std['test'],
                                                         uncertainty=False)


# In[10]:


# Plot heteroscedastic uncertainty.
plt.figure(3)
plt.plot(linex, au(linex), '-', lw=1, color='black')
plt.plot(train, train_error['rmse_all']-uncertainty_train.reshape(-1), 'o', alpha=0.2, color='black')
plt.plot(test, heteroscedastic_uncertainty['prediction'], 'g-', lw=1, alpha=0.4)
plt.xlabel('X')
plt.axis('tight')


# In[11]:


plt.show()

