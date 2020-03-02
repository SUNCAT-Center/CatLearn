#!/usr/bin/env python
# coding: utf-8

# # Voronoi Feature Generators <a name="head"></a>
# 
# In this tutorial, we will look at generating features from a database of organic donor-acceptor molecules from the [Computational Materials Repository](https://cmrdb.fysik.dtu.dk/?project=solar). This has been downloaded in the [ase-db](https://wiki.fysik.dtu.dk/ase/ase/db/db.html#module-ase.db) format so first off we load the atoms objects and get a target property. Then we convert the atoms objects into a feature array and test out a couple of different models.
# 
# This tutorial will give an indication of one way in which it is possible to handle atoms objects of different sizes. In particular, we focus on a feature set that scales with the number of atoms. We pad the feature vectors to a constant size to overcome this problem.
# 
# ## Table of Contents
# [(Back to top)](#head)
# 
# -   [Requirements](#requirements)
# -   [Data Setup](#data-setup)
# 
# ## Requirements <a name="requirements"></a>
# [(Back to top)](#head)
# 
# -   [CatLearn](https://github.com/SUNCAT-Center/CatLearn)
# -   [ASE](https://wiki.fysik.dtu.dk/ase/)
# -   [numpy](http://www.numpy.org/)
# 
# ## Data Setup <a name="data-setup"></a>
# [(Back to top)](#head)

# In[1]:


# Import packages.
import numpy as np
import ase.db
import random
from catlearn.fingerprint.voro import VoronoiFingerprintGenerator
from catlearn.preprocess.clean_data import clean_infinite, clean_variance


# We have stored our atomic structures in an ASE database file. Therefore, we first need to import it and put it in a list of atoms objects.

# In[2]:


# Connect the ase-db.
db = ase.db.connect('../../data/cubic_perovskites.db')
atoms = list(db.select(combination= 'ABO3'))[:10]
random.shuffle(atoms)

# Compile a list of atoms and target values.
alist = []
for row in atoms:
    try:
        alist.append(row.toatoms())
    except AttributeError:
        continue
print('pulled {} molecules from db'.format(len(alist)))


# In[3]:


# Check the size of the atomic strucures.
size = []
for a in alist:
    size.append(len(a))

print('min: {0}, mean: {1:.0f}, max: {2} atoms size'.format(
    min(size), sum(size)/len(size), max(size)))


# Now we generate the Voronoi fingerprints for our atoms objects. `voro.generate()` returns a Pandas dataframe.

# In[4]:


voro = VoronoiFingerprintGenerator(alist)
data_frame = voro.generate()


# In cases, where the generated featues does not apply to the input data, `NaN`s are returned.
# There are various ways of filling in this kind of data and the simplest is simply to remove features containing infinite values.
# 
# The conventional data format in CatLearn is a matrix, so we first convert the Pandas dataframe into a numpy array.

# In[5]:


matrix = data_frame.values
finite_numeric_data = clean_infinite(matrix)
print(np.shape(finite_numeric_data['train']))


# Furthermore, you might have data sets where certain features have completely the same value. Use `clean_variance` to get rid of those meaningless features.

# In[6]:


useful_data = clean_variance(finite_numeric_data['train'])
print(np.shape(useful_data['train']))


# We only selected the first 10 data points in this example, so there are likely to be some invariant features across those 10.
