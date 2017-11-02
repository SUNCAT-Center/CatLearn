"""Toy model to test out prediction routines."""
import numpy as np
import matplotlib.pyplot as plt

from atoml.preprocess.feature_preprocess import standardize
from atoml.regression import GaussianProcess

gradients = True
gradients_mode = 'Analytical'

def afunc(x):
    """S FUNCTION"""
    y = 20*((x-3)**(2))*(x-3) + 19
    #y = (x-3)**(2) + 1
    return y

def first_derivative(x):
    dydx =  60*(x-3)**2
    return dydx

test_points = 100

train = np.array([[1.5, 3.2, 4.1, 5.0]])
train = np.reshape(train, (np.shape(train)[1], np.shape(train)[0]))

target = []
for i in train:
    target.append(afunc(i))
target = np.array(target)
targetplot = target

if gradients == True:
    if gradients_mode == 'Analytical':
        gradients = []
        for i in train:
            gradients.append(first_derivative(i))
    if gradients_mode == 'Finite':
        delta = np.abs((train[0]-train[np.shape(train)[0]-1]) / 4.0)
        gradients = []
        for i in range(0,np.shape(train)[0]):
            gradients_finite = afunc(train[i]+delta)-afunc(train[i]-delta)
            gradients_finite = gradients_finite / 2*delta
            gradients.append(gradients_finite)
    if gradients_mode == 'Central':
        gradients = np.gradient(target[:,0],train[:,0])
    y_tilde = []
    # 1) Build y_tilde as:
    # y_tilde = y1, y2...yN, delta1, delta2...deltaN
    y_tilde = np.append(target, gradients)
    y_tilde = np.reshape(y_tilde,(np.shape(y_tilde)[0],1))
    target = y_tilde


linex = 1+4*np.random.random_sample((1, 1000))
linex = np.sort(linex)
liney = []
for i in linex:
    liney.append(afunc(i))

test = 1+4.0*np.random.random_sample((1, test_points))
test = np.sort(test)
test = np.reshape(test, (np.shape(test)[1], np.shape(test)[0]))

target = np.reshape(target,(np.shape(target)[1],np.shape(target)[0]))


# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian_gradients', 'width': 2.0, 'scaling': 1.0}}

gp = GaussianProcess(kernel_dict=kdict, regularization=1.0,
                     train_fp=train, train_target=target[0],
                     optimize_hyperparameters=True,standardize_target=False)

# Do the optimized predictions.
optimized = gp.predict(test_fp=test, uncertainty=True)

print(gp.theta_opt)

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(121)
ax.plot(linex[0], liney[0], '-', lw=1, color='black')
ax.plot(train, targetplot, 'o', alpha=0.2, color='black')
ax.plot(test, optimized['prediction'], 'g-', lw=1, alpha=0.4)
plt.xlabel('feature')
plt.ylabel('response')
plt.axis('tight')


plt.show()
