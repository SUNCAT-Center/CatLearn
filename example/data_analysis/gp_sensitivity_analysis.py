"""Script to test the prediction functions."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

from atoml.cross_validation import HierarchyValidation
from atoml.feature_preprocess import standardize
from atoml.predict import get_error
from atoml.gp_sensitivity import SensitivityAnalysis

ds = 500  # data size

# Define the hierarchey cv class method.
hv = HierarchyValidation(db_name='../../data/train_db.sqlite',
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

# Scale the data.
sfp = standardize(train_matrix=train_features, test_matrix=test_features)
train_features, test_features = sfp['train'], sfp['test']

# Test out a linear ridge regression model.
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
fr = reg.fit(train_features, train_targets)
pred = fr.predict(test_features)
err = get_error(prediction=pred, target=test_targets)
print('ridge prediction (rmse):', err['rmse_average'])

# Start the sensitivity analysis.
kdict = {'k1': {'type': 'gaussian', 'width': 10.}}
sen = SensitivityAnalysis(train_matrix=train_features,
                          train_targets=train_targets,
                          test_matrix=test_features, kernel_dict=kdict,
                          init_reg=0.001, init_width=10.)


sel = sen.backward_selection(predict=True, test_targets=test_targets,
                             selection=100)

# Plot the figures.
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(121)
x = range(np.shape(train_features)[1])

# Plot for all calculated feature sizes.
for i in sel:
    if i is not 'original':
        y = []
        for j in x:
            if j in sel[i]['abs_index']:
                k = np.argwhere(np.asarray(sel[i]['abs_index']) == j)[0][0]
                norms = (100. / np.sum(sel[i]['sensitivity'])) \
                    * sel[i]['sensitivity'][k]
                y.append(norms)
            else:
                y.append(0.)
        ax.plot(x, y, '.-', alpha=0.5)
    # Plot original feature space.
    if i is 'original':
        norms = (100. / np.sum(sel['original']['sensitivity'])) \
            * sel['original']['sensitivity']
        ax.plot(x, norms, '.-', color='black')

plt.xlabel('Feature No.')
plt.ylabel('Sensitivity (%)')

ax = fig.add_subplot(122)
pi, p = [], []
for i in sel:
    if i is not 'original':
        pi.append(i)
        p.append(sel[i]['prediction']['validation_error']['rmse_average'])
ax.plot(pi, p, 'o', color='black')
plt.xlabel('Feature Size')
plt.ylabel('Error (eV)')

plt.show()
