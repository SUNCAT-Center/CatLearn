# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 14:30:20 2016

@author: mhangaard

"""
from atoml.feature_preprocess import normalize, standardize
from atoml.cross_validation import k_fold
from atoml.predict import GaussianProcess
import numpy as np
from matplotlib import pyplot as plt
import random

nsplit = 3

data = np.genfromtxt('fpm.txt')

# Then we create the k-fold split
split_fpv_0 = []
split_energy = []
print('Creating', nsplit, '-fold split on survivors.')
split = k_fold(data, nsplit)
for i in range(nsplit):
    split_fpv_0.append(split[i][:, :-2])
    split_energy.append(split[i][:, -1])


split_fpv = list(split_fpv_0)

shape = np.shape(split_fpv_0[0])
print(shape)

forward = False
W = False
S = True
gk = True
lineark = False

plt.rc('text', usetex=False)
font = {'family': 'sans-serif',
        # 'sans-serif':, 'Helvetica',
        'style': 'normal',
        'stretch': 'normal'}
plt.rc('font', **font)
plt.rc('axes', linewidth=2)
plt.rc('figure')
plt.rc('lines', markeredgewidth=0, markersize=8, linewidth=2)
ticksize = 16
plt.rc('xtick', labelsize=ticksize+2)
plt.rc('ytick.major', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)
plt.rc('ytick', labelsize=ticksize+2)
plt.rc('ytick.major', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)

if forward:
    start = []
else:
    start = range(shape[1])

X = []
TRAIN_RMSE_trend = []
U_trend = []
TEST_trend = []
order = []
next_feature = random.randint(0, shape[1])  # random greedy start
for l in range(shape[1]):
    if forward:                                     # forward greedy
        start += [next_feature]
        FEATURES = range(shape[1])
        backward = False
        print(start)
    elif l == 0:
        FEATURES = range(1, shape[1])                # backward greedy
        backward = True
    elif l > 0:
        FEATURES.remove(next_feature)
    TRAIN_RMSE = []
    VAL_RMSE = []
    U = []
    for fs in FEATURES:
        if forward:
            indexes = start + [fs]      # forward greedy
        elif backward:
            indexes = list(FEATURES)    # backward greedy
            indexes.remove(fs)          # backward greedy
        # Subset of the fingerprint vector
        for i in range(nsplit):
            fpm = list(split_fpv_0)[i]
            shape = np.shape(fpm)
            reduced_fpv = split_fpv_0[i][:, indexes]
            split_fpv[i] = reduced_fpv
        # print('Make predictions based in k-fold samples')
        train_rmse = []
        val_rmse = []
        uncertainty = []
        for i in range(nsplit):
            # Setup the test, training and fingerprint datasets.
            traine = []
            train_fp = []
            testc = []
            teste = []
            test_fp = []
            for j in range(nsplit):
                if i != j:
                    for e in split_energy[j]:
                        traine.append(e)
                    for v in split_fpv[j]:
                        train_fp.append(v)
            for e in split_energy[i]:
                teste.append(e)
            for v in split_fpv[i]:
                test_fp.append(v)
            # Get the list of fingerprint vectors and normalize them.
            if S:
                nfp = standardize(train_matrix=train_fp, test_matrix=test_fp)
            else:
                nfp = normalize(train_matrix=train_fp, test_matrix=test_fp)
            # Do the training.
            # Optimize hyperparameters
            m = np.shape(nfp['train'])[1]
            sigma = np.ones(m)
            sigma *= .3
            regularization = .1
            theta = np.append(sigma, regularization)
            # Set up the prediction routine.
            kdict = {}
            if gk:
                kdict.update({'k1': {'type': 'gaussian', 'width': sigma}})
            if lineark:
                kdict.update({'k2': {'type': 'linear',
                                     # 'operation': 'multiplication'}})
                                     'const': 1.,
                                     }
                              })
            gp = GaussianProcess(train_fp=nfp['train'], train_target=traine,
                                 kernel_dict=kdict,
                                 regularization=regularization,
                                 optimize_hyperparameters=W)
            # Do the prediction
            pred = gp.get_predictions(test_fp=nfp['test'],
                                      get_validation_error=True,
                                      get_training_error=True,
                                      uncertainty=True,
                                      test_target=teste)
            # Print the error associated with the predictions.
            uncertainty.append(np.mean(pred['uncertainty']))
            train_rmse.append(pred['training_error']['absolute_average'])
            val_rmse.append(pred['validation_error']['absolute_average'])
        U.append(np.mean(uncertainty))
        TRAIN_RMSE.append(np.mean(train_rmse))
        VAL_RMSE.append(np.mean(val_rmse))
    # plt.scatter(FEATURES, U, c='r')
    # plt.scatter(FEATURES, TRAIN_RMSE, c='k')
    # plt.scatter(FEATURES, VAL_RMSE, c='b')
    if len(TRAIN_RMSE) == 0 and not forward:
        print(order)
        break
    min_train = np.argmin(TRAIN_RMSE)
    next_feature = FEATURES[min_train]
    lf = len(FEATURES)
    print('Min k-averaged training error:', TRAIN_RMSE[min_train])
    print('k-averaged uncertainty at best fit:', U[min_train])
    print('k-averaged validation at best fit:', VAL_RMSE[min_train])
    print('Best next feature:', next_feature, 'of', lf, 'features.')
    if lf < 30:
        print(FEATURES)
    TRAIN_RMSE_trend.append(TRAIN_RMSE[min_train])
    U_trend.append(U[min_train])
    TEST_trend.append(VAL_RMSE[min_train])
    if forward:
        X.append(l+1)
        if next_feature in indexes:
            print(indexes)
            print(sigma, regularization)
            break
    else:
        X.append(lf)
plt.plot(X, U_trend, c='r', marker='o', markersize=12,
         alpha=0.8, linewidth=3)
plt.plot(X, TRAIN_RMSE_trend, marker='o', c='k', markersize=12,
         alpha=0.8, linewidth=3)
plt.plot(X, TEST_trend, marker='o', c='b', markersize=12,
         alpha=0.8, linewidth=3)
plt.xlabel('Number of descriptors', fontsize=ticksize+2)
plt.ylabel('MAE (eV)', fontsize=ticksize+2)
fname = 'error_vs_ND'
if S:
    print('Standardized data.')
    fname += '_standardized'
else:
    print('Normalized data')
    fname += '_normalized'
if forward:
    fname += '_forward'
else:
    fname += '_backward'
if gk:
    fname += '_gaussk'
if lineark:
    fname += '_lineark'
plt.tight_layout(pad=0.6)
#plt.savefig(fname+'.pdf')
plt.show()
