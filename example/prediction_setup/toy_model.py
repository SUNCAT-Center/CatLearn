"""Toy model to test out prediction routines."""
import numpy as np
import matplotlib.pyplot as plt

from atoml.feature_preprocess import standardize
from atoml.predict import GaussianProcess


def afunc(x):
    """Define some polynomial function."""
    y = x - 50
    p = (y + 4) * (y + 4) * (y + 1) * (y - 1) * (y - 3.5) * (y - 2) * (y - 1)
    p += 40 * y + 80 * np.sin(10 * x)
    return 1 / 20 * p + 500


train_points = 30
test_points = 5000

train = 7.6 * np.random.random_sample((1, train_points)) - 4.2 + 50
target = []
for i in train:
    target.append(afunc(i))

nt = []
for i in range(train_points):
    nt.append(1.0*np.random.normal())
target += np.array(nt)

stdx = np.std(train)
stdy = np.std(target)
tstd = np.std(target, axis=1)

linex = 8 * np.random.random_sample((1, test_points)) - 4.5 + 50
linex = np.sort(linex)
liney = []
for i in linex:
    liney.append(afunc(i))

test = 8 * np.random.random_sample((1, test_points)) - 4.5 + 50
test = np.sort(test)

# Scale the input.
std = standardize(train_matrix=np.reshape(train, (np.shape(train)[1], 1)),
                  test_matrix=np.reshape(test, (np.shape(test)[1], 1)))

# Prediction parameters
sdt1 = np.sqrt(1e-1)
w1 = 0.95
sdt2 = np.sqrt(1e-5)
w2 = 0.142

# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': w1}}
gp = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                     train_fp=std['train'], train_target=target[0],
                     optimize_hyperparameters=False)

# Do the predictions.
pred = gp.get_predictions(test_fp=std['test'],
                          uncertainty=True)

upper = np.array(pred['prediction']) + (np.array(pred['uncertainty'] * tstd))
lower = np.array(pred['prediction']) - (np.array(pred['uncertainty'] * tstd))

# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': w2}}
gp = GaussianProcess(kernel_dict=kdict, regularization=sdt2**2,
                     train_fp=std['train'], train_target=target[0],
                     optimize_hyperparameters=False)

# Do the predictions.
over = gp.get_predictions(test_fp=std['test'],
                          uncertainty=True)

over_upper = np.array(over['prediction']) + \
 (np.array(over['uncertainty'] * tstd))
over_lower = np.array(over['prediction']) - \
 (np.array(over['uncertainty'] * tstd))

# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': [w1]}}
gp = GaussianProcess(kernel_dict=kdict, regularization=sdt1**2,
                     train_fp=std['train'], train_target=target[0],
                     optimize_hyperparameters=True)

# Do the predictions.
optp = gp.get_predictions(test_fp=std['test'],
                          uncertainty=True)

opt_upper = np.array(optp['prediction']) + \
 (np.array(optp['uncertainty'] * tstd))
opt_lower = np.array(optp['prediction']) - \
 (np.array(optp['uncertainty'] * tstd))

fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot(141)
ax.plot(linex[0], liney[0], '-', lw=1, color='black')
ax.plot(train[0], target[0], 'o', alpha=0.2, color='black')
ax.plot(test[0], pred['prediction'], 'b-', lw=1, alpha=0.4)
ax.fill_between(test[0], upper, lower, interpolate=True, color='blue',
                alpha=0.2)
plt.title('w: {0:.3f}, r: {1:.3f}'.format(w1 * stdx, sdt1 * stdy))
plt.xlabel('feature')
plt.ylabel('response')
plt.axis('tight')

ax = fig.add_subplot(142)
ax.plot(linex[0], liney[0], '-', lw=1, color='black')
ax.plot(train[0], target[0], 'o', alpha=0.2, color='black')
ax.plot(test[0], over['prediction'], 'r-', lw=1, alpha=0.4)
ax.fill_between(test[0], over_upper, over_lower, interpolate=True, color='red',
                alpha=0.2)
plt.title('w: {0:.3f}, r: {1:.3f}'.format(w2 * stdx, sdt2 * stdy))
plt.xlabel('feature')
plt.ylabel('response')
plt.axis('tight')

ax = fig.add_subplot(143)
ax.plot(linex[0], liney[0], '-', lw=1, color='black')
ax.plot(train[0], target[0], 'o', alpha=0.2, color='black')
ax.plot(test[0], optp['prediction'], 'g-', lw=1, alpha=0.4)
ax.fill_between(test[0], opt_upper, opt_lower, interpolate=True, color='green',
                alpha=0.2)
plt.title('w: {0:.3f}, r: {1:.3f}'.format(
    gp.kernel_dict['k1']['width'][0]*stdx, np.sqrt(gp.regularization)*stdy))
plt.xlabel('feature')
plt.ylabel('response')
plt.axis('tight')

ax = fig.add_subplot(144)
ax.plot(test[0], np.array(pred['uncertainty'] * tstd), '-', lw=1,
        color='blue')
ax.plot(test[0], np.array(over['uncertainty'] * tstd), '-', lw=1,
        color='red')
ax.plot(test[0], np.array(optp['uncertainty'] * tstd), '-', lw=1,
        color='green')
plt.title('Uncertainty Profile')
plt.xlabel('feature')
plt.ylabel('uncertainty')
plt.axis('tight')

plt.show()
