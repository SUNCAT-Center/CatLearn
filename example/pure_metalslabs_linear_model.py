# -*- coding: utf-8 -*-
""" Linear model of adsorption using known linear feature.

"""
from __future__ import print_function
import numpy as np
from os import listdir
from atoml.fingerprint_setup import return_fpv, get_combined_descriptors
from atoml.adsorbate_fingerprint import AdsorbateFingerprintGenerator
from atoml.db2thermo import db2mol, db2surf, mol2ref, get_refs, \
    get_formation_energies, db2surf_info
from atoml.predict import GaussianProcess
from atoml.feature_preprocess import standardize
import matplotlib.pyplot as plt

if 'pure_metals.txt' not in listdir('.'):
    fname = '../data/pure_metals.db'

    # Get dictionaries containing ab initio energies
    abinitio_energies, mol_dbids = db2mol('../data/mol.db', ['fmaxout<0.1',
                                                     'pw=500',
                                                     'vacuum=8',
                                                     'psp=gbrv1.5pbe'])
    abinitio_energies01, cand_dbids = db2surf(fname, ['series!=slab', 'phase=fcc'])
    abinitio_energies02, cand_dbids2 = db2surf(fname, ['series=slab', 'phase=fcc'])
    abinitio_energies.update(abinitio_energies01)
    abinitio_energies.update(abinitio_energies02)

    mol_dict = mol2ref(abinitio_energies)
    ref_dict = get_refs(abinitio_energies, mol_dict)

    # Calculate the formation energies and save in a dict
    formation_energies = get_formation_energies(abinitio_energies, ref_dict)

    # Get list of atoms objects
    train_cand = db2surf_info(fname, cand_dbids, formation_energies)
    print(len(train_cand), 'training examples.')

    # Get the adsorbate fingerprint class.
    fpv_train = AdsorbateFingerprintGenerator(bulkdb='../data/ref_bulks_k24.db')

    # Choose fingerprints.
    train_fpv = [
        # fpv_train.randomfpv,
        # fpv_train.primary_addatom,
        # fpv_train.primary_adds_nn,
        # fpv_train.Z_add,
        fpv_train.primary_adds_nn,
        fpv_train.adds_sum,
        fpv_train.primary_surfatom,
        fpv_train.primary_surf_nn,
        fpv_train.bulk,
        ]

    # Get names on the features
    L_F = get_combined_descriptors(train_fpv)
    print(len(L_F), 'Original descriptors:', L_F)

    # Generate fingerprints from atoms objects
    print('Getting fingerprint vectors.')
    cfpv = return_fpv(train_cand, train_fpv)
    print('Getting training taget values.')
    Ef = return_fpv(train_cand, fpv_train.info2Ef, use_prior=False)
    dbid = return_fpv(train_cand, fpv_train.get_dbid, use_prior=False)
    print(np.shape(Ef))
    assert np.isclose(len(L_F), np.shape(cfpv)[1])

    # Clean up the data
    print('Removing any training examples with NaN in the target value field')
    fpm0 = np.hstack([cfpv, np.vstack(Ef), np.vstack(dbid)])
    fpm_y = fpm0[~np.isnan(fpm0).any(axis=1)]

    # Save data matrix to txt file for later use
    print('Saving original', np.shape(fpm_y),
          'matrix. Last column is the ase.db id.' +
          ' Second last column is the target value.')
    np.savetxt('pure_metals.txt', fpm_y)
    y = fpm_y[:, -2]
    asedbid = fpm_y[:, -1]
else:
    fpm_y = np.genfromtxt('pure_metals.txt')
    y = fpm_y[:, -2]
    asedbid = fpm_y[:, -1]

# Separate database ids and target values from fingerprints
fpm = np.array(fpm_y[:, [3, 8]], ndmin=2)
# fpm_1d = np.array(fpm_y[:, [3]], ndmin=2)

gp_name = 'gp'

d0min = min(fpm[:, 0])
d0max = max(fpm[:, 0])
d1min = min(fpm[:, 1])
d1max = max(fpm[:, 1])
fpm_test = np.hstack([np.vstack(3 * list(np.linspace(d0min, d0max, 9))),
                      np.vstack(9*[7]+9*[8]+9*[9])])

nfp = standardize(train_matrix=fpm, test_matrix=fpm_test)
kdict = {
         'lk1': {'type': 'linear',
                 'const': .1,
                 'features': [0],
                 },
         #'gk': {'type': 'gaussian',
         #       'width': 1.0,
         #       'features': [1],
         #        'operation': 'multiplication'
         #       }
         }
# Run a Gaussian Process with a linear kernel

gp_name += '_SE'
#gp_name += '_LK'

gp = GaussianProcess(train_fp=nfp['train'], train_target=y,
                     kernel_dict=kdict,
                     regularization=1.,
                     optimize_hyperparameters=True)
# Do the training.
prediction = gp.get_predictions(test_fp=nfp['test'],
                                get_validation_error=False,
                                get_training_error=True)
print(gp.kernel_dict, gp.regularization)

plt.imshow(prediction['prediction'].reshape(3, 9),
           cmap='hot', interpolation='nearest', extent=[d0min,d0max,d1min,d1max])
plt.gcf().text(0.2, 0.8, 'RMSE = ' +
               str(round(prediction['training_error']['absolute_average'], 3)))
plt.colorbar()
plt.savefig(gp_name+'.pdf', format='pdf')
plt.show()
plt.clf()

# Make simple linear fits.
x6 = []
x7 = []
x8 = []
x9 = []
y6 = []
y7 = []
y8 = []
y9 = []
l6 = []
l7 = []
l8 = []
l9 = []
for X in range(len(fpm_y)):
    if fpm[X, 1] == 6:
        x6.append(fpm[X, 0])
        y6.append(y[X])
        l6.append(int(asedbid[X]))
    if fpm[X, 1] == 7:
        x7.append(fpm[X, 0])
        y7.append(y[X])
        l7.append(int(asedbid[X]))
    if fpm[X, 1] == 8:
        x8.append(fpm[X, 0])
        y8.append(y[X])
        l8.append(int(asedbid[X]))
    if fpm[X, 1] == 9:
        x9.append(fpm[X, 0])
        y9.append(y[X])
        l9.append(int(asedbid[X]))
xall = x6+x7+x8+x9
yall = y6+y7+y8+y9

plt.scatter(x7, y7, c='r')
for i7 in range(len(l7)):
    plt.annotate(l7[i7], xy=(x7[i7], y7[i7]))
plt.scatter(x8, y8, c='g')
for i8 in range(len(l8)):
    plt.annotate(l8[i8], xy=(x8[i8], y8[i8]))
plt.scatter(x9, y9, c='b')
for i9 in range(len(l9)):
    plt.annotate(l9[i9], xy=(x9[i9], y9[i9]))

start = plt.gca().get_xlim()[0]
end = plt.gca().get_xlim()[1]

a_all, c_all = np.polyfit(xall, yall, deg=1)
a7, c7 = np.polyfit(x7, y7, deg=1)
a8, c8 = np.polyfit(x8, y8, deg=1)
a9, c9 = np.polyfit(x9, y9, deg=1)

lx = np.linspace(start, end, 2)
lyall = a_all*lx + c_all
ly7 = a7*lx + c7
ly8 = a8*lx + c8
ly9 = a9*lx + c9
plt.plot(lx, ly7, alpha=0.6, c='r', label='7')
plt.plot(lx, ly8, alpha=0.6, c='g', label='8')
plt.plot(lx, ly9, alpha=0.6, c='b', label='9')

f_all = a_all*np.array(xall) + c_all
f7 = a7*np.array(x7) + c7
f8 = a8*np.array(x8) + c8
f9 = a9*np.array(x9) + c9

rmse7 = np.sum((f7 - np.array(y7))**2/len(x7))**(1/2.)
rmse8 = np.sum((f8 - np.array(y8))**2/len(x8))**(1/2.)
rmse9 = np.sum((f9 - np.array(y9))**2/len(x9))**(1/2.)
rmse = np.sum((f_all - np.array(yall))**2/len(xall))**(1/2.)

print(rmse)

plt.text(lx[-1], ly7[-1], 'RMSE = '+str(round(rmse7, 3)))
plt.text(lx[-1], ly8[-1], 'RMSE = '+str(round(rmse8, 3)))
plt.text(lx[-1], ly9[-1], 'RMSE = '+str(round(rmse9, 3)))
plt.legend(loc=2, borderaxespad=0.)
plt.savefig('linear_fit.pdf', format='pdf')
plt.show()

# Use a Gaussian Process for the same features.
