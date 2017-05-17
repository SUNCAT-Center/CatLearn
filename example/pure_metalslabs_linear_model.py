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
import matplotlib.pyplot as plt

if 'pure_metals.txt' not in listdir('.'):
    fname = 'pure_metals.db'

    # Get dictionaries containing ab initio energies
    abinitio_energies, mol_dbids = db2mol('mol.db', ['fmaxout<0.1',
                                                     'pw=500',
                                                     'vacuum=8',
                                                     'psp=gbrv1.5pbe'])
    abinitio_energies01, cand_dbids = db2surf(fname, ['series!=slab'])
    abinitio_energies02, cand_dbids2 = db2surf(fname, ['series=slab'])
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
    fpv_train = AdsorbateFingerprintGenerator(bulkdb='ref_bulks_k24.db')

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
    asedbid = fpm_y[-1]
else:
    fpm_y = np.genfromtxt('pure_metals.txt')
    y = fpm_y[:, -2]
    asedbid = fpm_y[-1]

# Separate database ids and target values from fingerprints
fpm = fpm_y[:, [3, 8]]
print(fpm)


# Make simple linear fits.
x6 = []
x7 = []
x9 = []
y6 = []
y7 = []
y9 = []
for X in range(len(fpm_y)):
    if fpm[X, 1] == 6:
        x6.append(fpm[X, 0])
        y6.append(y[X])
    if fpm[X, 1] == 7:
        x7.append(fpm[X, 0])
        y7.append(y[X])
    if fpm[X, 1] == 9:
        x9.append(fpm[X, 0])
        y9.append(y[X])

plt.scatter(x6, y6, c='r')
plt.scatter(x7, y7, c='g')
plt.scatter(x9, y9, c='b')

start = plt.gca().get_xlim()[0]
end = plt.gca().get_xlim()[1]
a7, c7 = np.polyfit(x7, y7, deg=1)
a9, c9 = np.polyfit(x9, y9, deg=1)

lx = np.linspace(start, end, 2)
ly7 = a7*lx + c7
ly9 = a9*lx + c9
plt.plot(lx, ly7, alpha=0.6, c='g')
plt.plot(lx, ly9, alpha=0.6, c='b')

f9 = a9*np.array(x9) + c9
f7 = a7*np.array(x7) + c7

rmse7 = np.sum((f7 - np.array(y7))**2/len(x7))**(1/2.)
rmse9 = np.sum((f9 - np.array(y9))**2/len(x9))**(1/2.)

rmse = np.sum((np.concatenate([f7, f9]) -
               np.concatenate([y7, y9]))**2/len(x9))**(1/2.)

print(rmse)

plt.text(lx[-1], ly7[-1], 'RMSE = '+str(round(rmse7, 3)))
plt.text(lx[-1], ly9[-1], 'RMSE = '+str(round(rmse9, 3)))
plt.show()

# Use a Gaussian Process for the same features.
