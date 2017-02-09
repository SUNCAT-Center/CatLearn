from __future__ import print_function

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from ase.ga.data import DataConnection
from atoml.data_setup import get_unique, get_train, target_standardize
from atoml.fingerprint_setup import standardize, normalize, return_fpv
from atoml.standard_fingerprint import StandardFingerprintGenerator
from atoml.particle_fingerprint import ParticleFingerprintGenerator
from atoml.predict import FitnessPrediction


db = DataConnection('gadb.db')

# Get all relaxed candidates from the db file.
all_cand = db.get_all_relaxed_candidates(use_extinct=False)

# Setup the test and training datasets.
testset = get_unique(candidates=all_cand, testsize=500, key='raw_score')
trainset = get_train(candidates=all_cand, trainsize=500,
                     taken_cand=testset['taken'], key='raw_score')

# Define fingerprint parameters.
sfpv = StandardFingerprintGenerator()
pfpv = ParticleFingerprintGenerator(get_nl=False, max_bonds=13)

# Get the list of fingerprint vectors and normalize them.
test_fp = return_fpv(testset['candidates'], [sfpv.eigenspectrum_fpv,
                                             pfpv.nearestneighbour_fpv],
                     use_prior=False)
train_fp = return_fpv(trainset['candidates'], [sfpv.eigenspectrum_fpv,
                                               pfpv.nearestneighbour_fpv],
                      use_prior=False)
nfp = normalize(train=train_fp, test=test_fp)
sfp = standardize(train=train_fp, test=test_fp)

ntrain = []
ntest = []
for i in nfp['train']:
    ntrain.append(i * 1.)
for i in nfp['test']:
    ntest.append(i * 1.)

nfp['train'] = ntrain
nfp['test'] = ntest

# Set up the prediction routine.
krr = FitnessPrediction(ktype='gaussian',
                        kwidth=0.5,
                        regularization=0.001)
cvm = krr.get_covariance(train_fp=nfp['train'])


def basis(descriptors):
    return descriptors * ([1] * len(descriptors))


pred = krr.get_predictions(train_fp=nfp['train'],
                           test_fp=nfp['test'],
                           cinv=cvm,
                           train_target=trainset['target'],
                           test_target=testset['target'],
                           get_validation_error=True,
                           get_training_error=True,
                           standardize_target=True,
                           uncertainty=True,
                           basis=basis)
ae = 0.
pe = 0.
for i, j, k in zip(pred['prediction'], pred['basis_analysis']['gX'],
                   testset['target']):
    ae += (i-k)**2
    pe += (j-k)**2
print((ae/len(testset['target']))**0.5, (pe/len(testset['target']))**0.5)
exit()

pe = []
st = target_standardize(testset['target'])
st = st['target']
sp = target_standardize(pred['prediction'])
sp = sp['target']
print('prediction | uncertainty | error')
for i, j, l, m, k in zip(pred['prediction'],
                         pred['uncertainty'],
                         pred['validation_rmse']['all'],
                         st, sp):
    e = (k - m) / j
    pe.append(e)
    print(i, '|', j, '|', l)
print('gaussian prediction:', pred['validation_rmse']['average'])
x = pd.Series(pe, name="Rel. Uncertainty")

print(min(pred['validation_rmse']['all']), max(pred['validation_rmse']['all']))

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
