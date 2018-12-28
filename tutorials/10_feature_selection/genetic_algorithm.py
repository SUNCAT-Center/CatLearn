#!/usr/bin/env python
# coding: utf-8

# # Feature Selection Genetic Algorithm
# 
# The principle behind the genetic algorithm for feature selection is relatively simple, the feature space is represented by a binary array. Features encoded with a one means that they are present in the optimized feature set, while a zero means they have been removed.
# 
# ## Setup

# In[1]:


import numpy as np

from ase.ga.data import DataConnection

from catlearn.api.ase_data_setup import get_unique, get_train
from catlearn.featurize.setup import FeatureGenerator
from catlearn.regression import GaussianProcess
from catlearn.preprocess.feature_engineering import single_transform
from catlearn.ga import GeneticAlgorithm


# ## Data Generation
# 
# To start with we import some data. For this tutorial, the data for alloyed nanoparticles are used.

# In[2]:


# Connect ase atoms database.
gadb = DataConnection('../../data/gadb.db')

# Get all relaxed candidates from the db file.
all_cand = gadb.get_all_relaxed_candidates(use_extinct=False)


# We then split this data into some training data and a holdout test set.

# In[3]:


testset = get_unique(atoms=all_cand, size=100, key='raw_score')

trainset = get_train(atoms=all_cand, size=500, taken=testset['taken'],
                     key='raw_score')

trainval = trainset['target']
testval = testset['target']


# Once the data is divided up, we then generate some feature sets. The eigenspectrum features are generated and then single transform engineering functions are used to expand the space slightly.

# In[4]:


generator = FeatureGenerator(atom_types=[78, 79], nprocs=1)
train_data = generator.return_vec(trainset['atoms'], [generator.eigenspectrum_vec])
test_data = generator.return_vec(testset['atoms'], [generator.eigenspectrum_vec])

train_data = single_transform(train_data)
test_data = single_transform(test_data)


# ## Baseline
# 
# Initially, a GP is trained on all the features and the error calculated as a baseline.

# In[5]:


kdict = [
    {
        'type': 'gaussian', 'width': 1., 'scaling': 1.,
        'dimension': 'single'
        }
    ]
gp = GaussianProcess(train_fp=train_data,
                     train_target=trainval,
                     kernel_list=kdict,
                     regularization=1e-2,
                     optimize_hyperparameters=True,
                     scale_data=True)

pred = gp.predict(test_fp=test_data, test_target=testval,
                  get_validation_error=True,
                  get_training_error=True)

score = pred['validation_error']['rmse_average']

print('all features: {0:.3f}'.format(score))


# ## Optimization
# 
# To optimize the feature set, a prediction function is first defined. This will take a boolean array from the genetic algorithm, transform the feature set and test against the holdout data. The error is then calculated and returned for that set of features. The genetic algorithm will aim to maximize the "fitness" of a population of candidates; therefore, the negative of the average cost is returned in this example.

# In[6]:


def fitf(train_features, train_targets, test_features, test_targets):
    """Define the fitness function for the GA."""
    kdict = [
        {
            'type': 'gaussian', 'width': 1.,
            'scaling': 1., 'dimension': 'single'
            }
        ]
    gp = GaussianProcess(train_fp=train_features,
                         train_target=train_targets,
                         kernel_list=kdict,
                         regularization=1e-2,
                         optimize_hyperparameters=True,
                         scale_data=True)

    pred = gp.predict(test_fp=test_features,
                      test_target=test_targets,
                      get_validation_error=True)

    score = pred['validation_error']['rmse_average']

    return -score


# Then the search can be run. The population size is set to 10 candidates and the number of dimensions equal to the total number of features.

# In[ ]:


ga = GeneticAlgorithm(population_size=3,
                      fit_func=fitf,
                      features=train_data,
                      targets=trainval, 
                      population=None,
                      accuracy=5)

ga.search(500, natural_selection=False, verbose=True, repeat=3)


# Once the search has finished, there will be a number of useful attributes attached to the class. The `ga.search()` function doesn't return anything, so this is the way we access the results. To start with we can look at the fitness for the final population.

# In[8]:


print(ga.fitness)


# The fitnesses returned are ordered from best to worst, corresponding to the same order at the results in the `population` attribute. This can be accessed and converted to a boolean array as follows.

# In[9]:


print(np.array(ga.population, dtype=np.bool))


# This can then be used to acquire the optimal set of features for further iterations of the model.

# In[10]:


final_pop = np.array(ga.population, dtype=np.bool)

optimized_train = train_data[:, final_pop[0]]
optimized_test = test_data[:, final_pop[0]]

print(np.shape(optimized_train), np.shape(optimized_test))


# ## Conclusions
# 
# It appears as though the genetic algorithm can aid in finding good feature sets upon which to train the model.
