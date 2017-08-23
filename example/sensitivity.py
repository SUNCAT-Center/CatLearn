"""Script to test the prediction functions."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

from atoml.cross_validation import HierarchyValidation
from atoml.feature_preprocess import standardize
from atoml.predict import GaussianProcess, get_error
from atoml.sensitivity import mean_sensitivity

ds = 500
r = 50

# Define the hierarchey cv class method.
hv = HierarchyValidation(db_name='../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='split')
# Split the data into subsets.
hv.split_index(min_split=ds, max_split=ds*2)
# Load data back in from save file.
ind = hv.load_split()

# Split out the various data.
train_features = np.array(hv._compile_split(ind['1_1'])[:, 1:-1], np.float64)
train_targets = np.array(hv._compile_split(ind['1_1'])[:, -1:], np.float64)
test_features = np.array(hv._compile_split(ind['1_2'])[:, 1:-1], np.float64)
test_targets = np.array(hv._compile_split(ind['1_2'])[:, -1:], np.float64)

train_targets = train_targets.reshape(len(train_targets), )
test_targets = test_targets.reshape(len(test_targets), )

sfp = standardize(train_matrix=train_features, test_matrix=test_features)
train_features, test_features = sfp['train'], sfp['test']

reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
fr = reg.fit(train_features, train_targets)
pred = fr.predict(test_features)

err = get_error(prediction=pred, target=test_targets)
print('ridge prediction (rmse):', err['rmse_average'])

kdict = {'k1': {'type': 'gaussian', 'width': 10.}}
gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                     kernel_dict=kdict, regularization=0.001,
                     optimize_hyperparameters=True)

pred = gp.get_predictions(test_fp=test_features,
                          test_target=test_targets,
                          get_validation_error=True,
                          get_training_error=True)

base = pred['validation_error']['rmse_average']
print('gaussian prediction (rmse):', pred['validation_error']['rmse_average'])

width = gp.kernel_dict['k1']['width']
reg = gp.regularization

print('Starting sensitivity analysis')
sv = mean_sensitivity(train_matrix=train_features,
                      train_targets=train_targets,
                      test_matrix=test_features, kernel_dict=kdict,
                      regularization=reg)
print('Finish sensitivity analysis')

ind = list(range(len(sv)))

# Plot the figure.
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(121)
ax.plot(ind, sv, 'o', color='black')
plt.xlabel('Feature No.')
plt.ylabel('Sensitivity')

sort_list = [list(i) for i in zip(*sorted(zip(sv, ind), key=lambda x: x[0],
                                          reverse=True))]

train_store, test_store = train_features, test_features

p = []
pi = list(range(r))
for i in range(r):
    train_features = np.delete(train_store, sort_list[1][i:], 1)
    test_features = np.delete(test_store, sort_list[1][i:], 1)
    print(np.shape(train_features))

    kdict = {'k1': {'type': 'gaussian', 'width': 10.}}
    gp = GaussianProcess(train_fp=train_features, train_target=train_targets,
                         kernel_dict=kdict, regularization=0.001,
                         optimize_hyperparameters=True)

    pred = gp.get_predictions(test_fp=test_features,
                              test_target=test_targets,
                              get_validation_error=True,
                              get_training_error=True)
    p.append(pred['validation_error']['rmse_average'])

    print('gaussian prediction (rmse):',
          pred['validation_error']['rmse_average'])

# Plot the figure.
ax = fig.add_subplot(122)
ax.plot(pi, p, 'o', color='black')
plt.xlabel('Feature Size')
plt.ylabel('Error')

plt.show()
