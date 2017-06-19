"""f(x) = 1/20 (x + 4)(x + 2)(x + 1)(x − 1)(x − 3)+ 2"""
import numpy as np
import matplotlib.pyplot as plt

from atoml.feature_preprocess import standardize
from atoml.predict import GaussianProcess


def afunc(x):
    p = (x + 4) * (x + 4) * (x + 1) * (x - 1) * (x - 3.5) * (x - 2) * (x - 1)
    return 1 / 20 * p + 24


train_points = 10
test_points = 5000

train = 8 * np.random.random_sample((1, train_points)) - 4.5
target = []
for i in train:
    target.append(afunc(i))
# Add noise.
# target = 2 * target + np.random.random_sample((1, train_points)) - 0.5
tstd = np.std(target, axis=1)

linex = 8 * np.random.random_sample((1, test_points)) - 4.5
linex = np.sort(linex)
liney = []
for i in linex:
    liney.append(afunc(i))

test = 8 * np.random.random_sample((1, test_points)) - 4.5
test = np.sort(test)

# Scale the input.
std = standardize(train_matrix=np.reshape(train, (np.shape(train)[1], 1)),
                  test_matrix=np.reshape(test, (np.shape(test)[1], 1)))

# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': 1.}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.1)

# Do the predictions.
pred = gp.get_predictions(train_fp=std['train'],
                          test_fp=std['test'],
                          train_target=target[0],
                          uncertainty=True,
                          optimize_hyperparameters=False)

upper = np.array(pred['prediction']) + (np.array(pred['uncertainty'] * tstd))
lower = np.array(pred['prediction']) - (np.array(pred['uncertainty'] * tstd))

# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': 0.1}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)

# Do the predictions.
over = gp.get_predictions(train_fp=std['train'],
                          test_fp=std['test'],
                          train_target=target[0],
                          uncertainty=True,
                          optimize_hyperparameters=False)

over_upper = np.array(over['prediction']) + \
 (np.array(over['uncertainty'] * tstd))
over_lower = np.array(over['prediction']) - \
 (np.array(over['uncertainty'] * tstd))

# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': 0.1}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)

# Do the predictions.
optp = gp.get_predictions(train_fp=std['train'],
                          test_fp=std['test'],
                          train_target=target[0],
                          uncertainty=True,
                          optimize_hyperparameters=True)

opt_upper = np.array(optp['prediction']) + \
 (np.array(optp['uncertainty'] * tstd))
opt_lower = np.array(optp['prediction']) - \
 (np.array(optp['uncertainty'] * tstd))

fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot(131)
ax.plot(linex[0], liney[0], '-', lw=1, color='black')
ax.plot(train[0], target[0], 'o', alpha=0.6, color='black')
ax.plot(test[0], pred['prediction'], 'b-', lw=1, alpha=0.4)
ax.fill_between(test[0], upper, lower, interpolate=True, color='blue',
                alpha=0.2)
plt.title('w: 1.00, r: 0.100')
plt.xlabel('feature')
plt.ylabel('response')
plt.axis('tight')

ax = fig.add_subplot(132)
ax.plot(linex[0], liney[0], '-', lw=1, color='black')
ax.plot(train[0], target[0], 'o', alpha=0.6, color='black')
ax.plot(test[0], over['prediction'], 'r-', lw=1, alpha=0.4)
ax.fill_between(test[0], over_upper, over_lower, interpolate=True, color='red',
                alpha=0.2)
plt.title('w: 0.10, r: 0.001')
plt.xlabel('feature')
plt.ylabel('response')
plt.axis('tight')

ax = fig.add_subplot(133)
ax.plot(linex[0], liney[0], '-', lw=1, color='black')
ax.plot(train[0], target[0], 'o', alpha=0.6, color='black')
ax.plot(test[0], optp['prediction'], 'g-', lw=1, alpha=0.4)
ax.fill_between(test[0], opt_upper, opt_lower, interpolate=True, color='green',
                alpha=0.2)
plt.title('w: {0:.2f}, r: {0:.3f}'.format(
    optp['optimized_kernels']['k1']['width'][0],
    optp['optimized_regularization']))
plt.xlabel('feature')
plt.ylabel('response')
plt.axis('tight')

plt.show()
