"""Small data with GP regression timings."""
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF

from atoml.regression import GaussianProcess

rng = np.random.RandomState(0)

# Generate sample data
X = 15 * rng.rand(200, 1)
y = np.sin(X).ravel()
y += 3 * (0.5 - rng.rand(X.shape[0]))  # add noise

gp_kernel = 1. * RBF(length_scale=0.5) + WhiteKernel(1e-1)
gpr = GaussianProcessRegressor(kernel=gp_kernel)
stime = time.time()
gpr.fit(X, y)
print("Time for sklearn fitting: %.3f" % (time.time() - stime))

X_plot = np.linspace(0, 20, 10000)[:, None]
stime = time.time()
y_gpr = gpr.predict(X_plot, return_std=False)
print("Time for sklearn prediction: %.3f" % (time.time() - stime))

kdict = {
    'k1': {'type': 'gaussian', 'width': [0.5]},
}

stime = time.time()
gp = GaussianProcess(
    kernel_dict=kdict, regularization=1e-1, train_fp=X, train_target=y,
    optimize_hyperparameters=True, scale_data=False)
print("Time for atoml fitting: %.3f" % (time.time() - stime))

stime = time.time()
y_atoml = gp.predict(test_fp=X_plot, uncertainty=True)
print("Time for atoml prediction: %.3f" % (time.time() - stime))
y_atoml = y_atoml['prediction']

# Plot results
plt.figure(figsize=(10, 5))
lw = 2
plt.scatter(X, y, c='k', label='data')
plt.plot(X_plot, np.sin(X_plot), color='navy', lw=lw, label='True')
plt.plot(X_plot, y_gpr, color='turquoise', lw=lw,
         label='sklearn')
plt.plot(X_plot, y_atoml, color='darkorange', lw=lw,
         label='atoml')
plt.xlabel('data')
plt.ylabel('target')
plt.xlim(0, 20)
plt.ylim(-4, 4)
plt.title('scikit-learn vs AtoML')
plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
plt.show()
