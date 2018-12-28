#!/usr/bin/env python
# coding: utf-8

# # Hierarch Cross-Validation <a name="head"></a>
#
# This tutorial will go through setting up a function to perform a hierarchy of cross-validation. This will ultimately allow us to create a learning curve and investigate the learning rate of the model.
#
# ## Table of Contents
# [(Back to top)](#head)
#
# -   [Data Setup](#data-setup)
# -   [Prediction Setup](#prediction-setup)
# -   [Cross-valisation Setup](#cross-validation-setup)
# -   [Prediction Analysis](#prediction-analysis)
# -   [Conclusions](#conclusions)
#
# ## Data Setup <a name="data-setup"></a>
# [(Back to top)](#head)
#
# First, we need to import some functions.

# In[1]:


# Comment out this line when exported to .py file
# get_ipython().run_line_magic('matplotlib', 'inline')

import os
import numpy as np
import matplotlib.pyplot as plt

from ase.ga.data import DataConnection

from catlearn.featurize.setup import FeatureGenerator
from catlearn.cross_validation import Hierarchy
from catlearn.regression import RidgeRegression, GaussianProcess
from catlearn.regression.cost_function import get_error


# Then we can load some data. There is some pre-generated data in an ase-db so first, the atoms objects are loaded into a list. Then they are fed through a feature generator.

# In[2]:


# Connect ase atoms database.
gadb = DataConnection('../../data/gadb.db')

# Get all relaxed candidates from the db file.
all_cand = gadb.get_all_relaxed_candidates(use_extinct=False)
print('Loaded {} atoms objects'.format(len(all_cand)))

# Generate the feature matrix.
fgen = FeatureGenerator()
features = fgen.return_vec(all_cand, [fgen.eigenspectrum_vec])
print('Generated {} feature matrix'.format(np.shape(features)))

# Get the target values.
targets = []
for a in all_cand:
    targets.append(a.info['key_value_pairs']['raw_score'])
print('Generated {} target vector'.format(np.shape(targets)))


# It is important to note that the `all_cand` variable is simply a list of atoms objects. There are no constraints on how this should be set up, the above example is just a succinct method for generating the list.
#
# ## Prediction Setup <a name="prediction-setup"></a>
# [(Back to top)](#head)
#
# Once the feature matrix and target vector have been generated we can define a prediction function. This will be called on all subsets of data and is expected to take test and training features and targets. The function should return a dictionary with `{'result': list, 'size': list}`. The `result` will typically be an average error and the `size` will be the number of training data points. The first prediction routine that we define utilizes ridge regression.

# In[3]:


def rr_predict(train_features, train_targets, test_features, test_targets):
    """Function to perform the RR predictions."""
    data = {}

    # Set up the ridge regression function.
    rr = RidgeRegression(W2=None, Vh=None, cv='loocv')
    b = rr.find_optimal_regularization(X=train_features, Y=train_targets)
    coef = rr.RR(X=train_features, Y=train_targets, omega2=b)[0]

    # Test the model.
    sumd = 0.
    err = []
    for tf, tt in zip(test_features, test_targets):
        p = np.dot(coef, tf)
        sumd += (p - tt) ** 2
        e = ((p - tt) ** 2) ** 0.5
        err.append(e)
    error = (sumd / len(test_features)) ** 0.5

    data['result'] = error
    data['size'] = len(train_targets)

    return data


# We can define any prediction routine in this format. The following provides a second example with Gaussian process predictions.

# In[4]:


def gp_predict(train_features, train_targets, test_features, test_targets):
    """Function to perform the GP predictions."""
    data = {}

    kdict = [
        {'type': 'gaussian', 'width': 1., 'scaling': 1., 'dimension': 'single'},
        ]
    gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                         kernel_list=kdict, regularization=1e-2,
                         optimize_hyperparameters=True, scale_data=True)

    pred = gp.predict(test_fp=test_features)

    data['result'] = get_error(pred['prediction'],
                               test_targets)['rmse_average']
    data['size'] = len(train_targets)

    return data


# ## Cross-validation Setup <a name="cross-validation-setup"></a>
# [(Back to top)](#head)
#
# Next, we can run the cross-validation on the generated data. In order to allow for flexible storage of large numbers of data subsets, we convert the feature and target arrays to a simple db format. This is performed with the `todb()` function. After this, we split up the db index to define the subsets of data with the `split_index()` function. In this case, the maximum amount of data considered in 1000 data points and the smallest set of data will contain a minimum of 50 data points.

# In[5]:


# Initialize hierarchy cv class.
hv = Hierarchy(db_name='test.sqlite', file_name='hierarchy')
# Convert features and targets to simple db format.
hv.todb(features=features, targets=targets)
# Split the data into subsets.
ind = hv.split_index(min_split=50, max_split=1000)


# ## Prediction Analysis <a name="prediction-analysis"></a>
# [(Back to top)](#head)
#
# The analysis is first performed with ridge regression. Predictions are made for all subsets of data and the averaged errors plotted against the data size. What is typically observed is that as the size of the data subset increases the error decreases.

# In[ ]:


# Make the predictions for each subset.
pred = hv.split_predict(index_split=ind, predict=rr_predict)

# Get mean error at each data size.
means, meane = hv.transform_output(pred)

# Plot the results.
plt.figure(0)
plt.plot(pred[1], pred[0], 'o', c='b', alpha=0.5)
plt.plot(means, meane, '-', alpha=0.9, c='black')


# We can perform the same analysis with the Gaussian process predictions.

# In[ ]:


# Make the predictions for each subset.
pred = hv.split_predict(index_split=ind, predict=gp_predict)

# Get mean error at each data size.
means, meane = hv.transform_output(pred)

# Plot the results.
plt.figure(1)
plt.plot(pred[1], pred[0], 'o', c='r', alpha=0.5)
plt.plot(means, meane, '-', alpha=0.9, c='black')


# We can then clean up the directory and remove saved files.

# In[ ]:


# Remove output.
os.remove('hierarchy.pickle')
os.remove('test.sqlite')


# ## Conclusions <a name="conclusions"></a>
# [(Back to top)](#head)
#
# This tutorial has gone through generating a simple set of functions to analyze the data size effect on prediction accuracy for two different models.
#

# In[ ]:


plt.show()
