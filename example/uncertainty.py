"""Example to get predictive uncertainty."""
from __future__ import print_function
from __future__ import absolute_import

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from ase.ga.data import DataConnection
from atoml.data_setup import get_unique, get_train
from atoml.fingerprint_setup import return_fpv
from atoml.standard_fingerprint import StandardFingerprintGenerator
from atoml.particle_fingerprint import ParticleFingerprintGenerator
from atoml.feature_preprocess import normalize
from atoml.predict import GaussianProcess, target_standardize

db = DataConnection('gadb.db')

# Get all relaxed candidates from the db file.
all_cand = db.get_all_relaxed_candidates(use_extinct=False)

# Setup the test and training datasets.
testset = get_unique(atoms=all_cand, size=500, key='raw_score')
trainset = get_train(atoms=all_cand, size=500, taken=testset['taken'],
                     key='raw_score')

# Define fingerprint parameters.
sfpv = StandardFingerprintGenerator()
pfpv = ParticleFingerprintGenerator(get_nl=False, max_bonds=13)

# Get the list of fingerprint vectors and normalize them.
test_fp = return_fpv(testset['atoms'], [sfpv.eigenspectrum_fpv,
                                        pfpv.nearestneighbour_fpv],
                     use_prior=False)
train_fp = return_fpv(trainset['atoms'], [sfpv.eigenspectrum_fpv,
                                          pfpv.nearestneighbour_fpv],
                      use_prior=False)
nfp = normalize(train_matrix=train_fp, test_matrix=test_fp)

# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': 0.5}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)


def basis(descriptors):
    """Define simple linear basis."""
    return descriptors * ([1] * len(descriptors))


pred = gp.get_predictions(train_fp=nfp['train'],
                          test_fp=nfp['test'],
                          train_target=trainset['target'],
                          test_target=testset['target'],
                          get_validation_error=True,
                          get_training_error=True,
                          standardize_target=True,
                          uncertainty=True,
                          basis=basis,
                          optimize_hyperparameters=False)

print('GP:', pred['validation_rmse']['average'], 'Residual:',
      pred['basis_analysis']['validation_rmse']['average'])

pe = []
st = target_standardize(testset['target'])
st = st['target']
sp = target_standardize(pred['prediction'])
sp = sp['target']
for i, j, l, m, k in zip(pred['prediction'],
                         pred['uncertainty'],
                         pred['validation_rmse']['all'],
                         st, sp):
    e = (k - m) / j
    pe.append(e)
x = pd.Series(pe, name="Rel. Uncertainty")

pred['Prediction'] = pred['prediction']
pred['Actual'] = testset['target']
pred['Error'] = pred['validation_rmse']['all']
pred['Uncertainty'] = pred['uncertainty']
index = [i for i in range(len(test_fp))]
df = pd.DataFrame(data=pred, index=index)
with sns.axes_style("white"):
    plt.subplot(311)
    plt.title('Validation RMSE: {0:.3f}'.format(
        pred['validation_rmse']['average']))
    sns.regplot(x='Actual', y='Prediction', data=df)
    plt.subplot(312)
    ax = sns.regplot(x='Uncertainty', y='Error', data=df, fit_reg=False)
    ax.set_ylim([min(pred['validation_rmse']['all']),
                 max(pred['validation_rmse']['all'])])
    ax.set_yscale('log')
    plt.subplot(313)
    sns.distplot(x, bins=50, kde=False)

plt.show()
