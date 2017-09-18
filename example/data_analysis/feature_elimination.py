"""Script to test descriptors for the ML model."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
import time

from atoml.cross_validation import Hierarchy
from atoml.feature_preprocess import standardize
from atoml.feature_engineering import (single_transform, get_order_2,
                                       get_order_2ab, get_ablog,
                                       get_div_order_2)
from atoml.feature_elimination import FeatureScreening
from atoml.utilities import clean_variance

# Set some parameters.
plot = False
ds = 100
c = 'pearson'

# Define the hierarchey cv class method.
hv = Hierarchy(db_name='../../data/train_db.sqlite', table='FingerVector',
               file_name='split')
# Split the data into subsets.
hv.split_index(min_split=ds, max_split=ds*2)
# Load data back in from save file.
ind = hv.load_split()

# Split out the various data.
train_data = np.array(hv._compile_split(ind['1_1'])[:, 1:-1], np.float64)
train_target = np.array(hv._compile_split(ind['1_1'])[:, -1:], np.float64)
test_data = np.array(hv._compile_split(ind['1_2'])[:, 1:-1], np.float64)
test_target = np.array(hv._compile_split(ind['1_2'])[:, -1:], np.float64)

cv = clean_variance(train=train_data, test=test_data)
train_data, test_data = cv['train'], cv['test']

d, f = np.shape(train_data)

# Scale and shape the data.
std = standardize(train_matrix=train_data, test_matrix=test_data)
train_data, test_data = std['train'], std['test']
train_target = train_target.reshape(len(train_target), )
test_target = test_target.reshape(len(test_target), )

# Expand feature space to add single variable transforms.
test_data1 = single_transform(test_data)
train_data1 = single_transform(train_data)
# test_data2 = get_order_2(test_data)
# train_data2 = get_order_2(train_data)
# test_data3 = get_div_order_2(test_data)
# train_data3 = get_div_order_2(train_data)
# test_data4 = get_order_2ab(test_data, 2, 3)
# train_data4 = get_order_2ab(train_data, 2, 3)
# test_data5 = get_ablog(test_data, 2, 3)
# train_data5 = get_ablog(train_data, 2, 3)

test_data = np.concatenate((test_data, test_data1), axis=1)
train_data = np.concatenate((train_data, train_data1), axis=1)
# test_data = np.concatenate((test_data, test_data2), axis=1)
# train_data = np.concatenate((train_data, train_data2), axis=1)
# test_data = np.concatenate((test_data, test_data3), axis=1)
# train_data = np.concatenate((train_data, train_data3), axis=1)
# test_data = np.concatenate((test_data, test_data4), axis=1)
# train_data = np.concatenate((train_data, train_data4), axis=1)
# test_data = np.concatenate((test_data, test_data5), axis=1)
# train_data = np.concatenate((train_data, train_data5), axis=1)

if plot:
    # Get descriptor correlation
    corrt = []
    for cd in np.transpose(train_data):
        if c is 'pearson':
            corrt.append(pearsonr(cd, train_target)[0])
        elif c is 'spearman':
            corrt.append(spearmanr(cd, train_target)[0])
        elif c is 'kendall':
            corrt.append(kendalltau(cd, train_target)[0])
    ind = list(range(len(corrt)))

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(131)
    ax.plot(ind, np.abs(corrt), '-', color='black', lw=0.5)
    ax.axis([0, len(ind), 0, 1])
    plt.title('Original Features')
    plt.xlabel('Feature No.')
    plt.ylabel('Correlation')

print('\nElimination based on %s correlation\n' % c)
screen = FeatureScreening(correlation=c, iterative=False)
st = time.time()
features = screen.eliminate_features(target=train_target,
                                     train_features=train_data,
                                     test_features=test_data,
                                     size=d, step=None, order=None)
print('screening took:', time.time() - st, 'for', np.shape(train_data))

if plot:
    corrt = []
    for cd in np.transpose(features[0]):
        if c is 'pearson':
            corrt.append(pearsonr(cd, train_target)[0])
        elif c is 'spearman':
            corrt.append(spearmanr(cd, train_target)[0])
        elif c is 'kendall':
            corrt.append(kendalltau(cd, train_target)[0])
    ind = list(range(len(corrt)))

    ax = fig.add_subplot(132)
    ax.plot(ind, np.abs(corrt), '-', color='black', lw=0.5)
    ax.axis([0, len(ind), 0, 1])
    plt.title('Screening Features')
    plt.xlabel('Feature No.')
    plt.ylabel('Correlation')

screen = FeatureScreening(correlation=c, iterative=True,
                          regression='ridge')
st = time.time()
features = screen.eliminate_features(target=train_target,
                                     train_features=train_data,
                                     test_features=test_data,
                                     size=d, step=None, order=None)
print('iterative took:', time.time() - st, 'for', np.shape(train_data))

if plot:
    corrt = []
    for cd in np.transpose(features[0]):
        if c is 'pearson':
            corrt.append(pearsonr(cd, train_target)[0])
        elif c is 'spearman':
            corrt.append(spearmanr(cd, train_target)[0])
        elif c is 'kendall':
            corrt.append(kendalltau(cd, train_target)[0])
    ind = list(range(len(corrt)))

    ax = fig.add_subplot(133)
    ax.plot(ind, np.abs(corrt), '-', color='black', lw=0.5)
    ax.axis([0, len(ind), 0, 1])
    plt.title('Iterative Features')
    plt.xlabel('Feature No.')
    plt.ylabel('Correlation')

    plt.show()
