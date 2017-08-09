"""Basic test for the manifold model."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.decomposition import PCA
from sklearn import manifold

from atoml.utilities import clean_variance
from atoml.cross_validation import HierarchyValidation
from atoml.feature_preprocess import standardize
from atoml.feature_engineering import single_transform
from atoml.predict import GaussianProcess

clean = True  # Clean zero-varience features
expand = False
new_data = True

nc = 5


def do_predict(train, test, train_target, test_target, hopt=False):
    """Function to make predictions."""
    std = standardize(train_matrix=train, test_matrix=test)

    kdict = {'k1': {'type': 'gaussian', 'width': 1.}}
    gp = GaussianProcess(train_fp=std['train'], train_target=train_target,
                         kernel_dict=kdict, regularization=1e-6,
                         optimize_hyperparameters=hopt)

    pred = gp.get_predictions(test_fp=std['test'],
                              test_target=test_target,
                              get_validation_error=True,
                              get_training_error=True)

    return pred


# Define the hierarchey cv class method.
hv = HierarchyValidation(db_name='../../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='split')
# Split the data into subsets.
if new_data:
    hv.split_index(min_split=50, max_split=1000)
# Load data back in from save file.
ind = hv.load_split()
fig = plt.figure(figsize=(15, 8))

train_data = np.array(hv._compile_split(ind['1_1'])[:, 1:-1], np.float64)
train_target = np.array(hv._compile_split(ind['1_1'])[:, -1:], np.float64)

test_data = np.array(hv._compile_split(ind['1_2'])[:, 1:-1], np.float64)
test_target = np.array(hv._compile_split(ind['1_2'])[:, -1:], np.float64)
d, f = np.shape(train_data)
data = np.concatenate((train_data, test_data), axis=0)
if clean:
    train_data = clean_variance(data)['train']
if expand:
    # Expand feature space to add single variable transforms.
    train_data = np.concatenate((data, single_transform(data)),
                                axis=1)

ax = fig.add_subplot(151)
pca = PCA(n_components=nc)
model = pca.fit(X=data[:d, :])
Y_data = model.transform(data)
plt.scatter(Y_data[:d, 0], Y_data[:d, 1], c=train_target, cmap=plt.cm.Spectral,
            alpha=0.5)
plt.title("PCA")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

print('PCA')
a = do_predict(train=Y_data[:d, :], test=Y_data[d:, :],
               train_target=train_target, test_target=test_target, hopt=True)

# Print the error associated with the predictions.
print('Training error:', a['training_error']['rmse_average'])
print('Model error:', a['validation_error']['rmse_average'])


ax = fig.add_subplot(152)
iso = manifold.Isomap(n_components=nc, n_neighbors=20)
fitted = iso.fit(data[:d, :])
Y_data = fitted.fit_transform(data)
plt.scatter(Y_data[:d, 0], Y_data[:d, 1], c=train_target, cmap=plt.cm.Spectral,
            alpha=0.5)
plt.title("Isomap")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

print('Isomap')
a = do_predict(train=Y_data[:d, :], test=Y_data[d:, :],
               train_target=train_target, test_target=test_target, hopt=True)

# Print the error associated with the predictions.
print('Training error:', a['training_error']['rmse_average'])
print('Model error:', a['validation_error']['rmse_average'])

ax = fig.add_subplot(153)
mds = manifold.MDS(n_components=nc)
fitted = mds.fit(data[:d, :])
Y_data = fitted.fit_transform(data)
plt.scatter(Y_data[:d, 0], Y_data[:d, 1], c=train_target, cmap=plt.cm.Spectral,
            alpha=0.5)
plt.title("MDS")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

print('MDS')
a = do_predict(train=Y_data[:d, :], test=Y_data[d:, :],
               train_target=train_target, test_target=test_target, hopt=True)

# Print the error associated with the predictions.
print('Training error:', a['training_error']['rmse_average'])
print('Model error:', a['validation_error']['rmse_average'])

ax = fig.add_subplot(154)
se = manifold.SpectralEmbedding(n_components=nc, n_neighbors=20)
fitted = se.fit(data[:d, :])
Y_data = fitted.fit_transform(data)
plt.scatter(Y_data[:d, 0], Y_data[:d, 1], c=train_target, cmap=plt.cm.Spectral,
            alpha=0.5)
plt.title("SpectralEmbedding")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

print('SpectralEmbedding')
a = do_predict(train=Y_data[:d, :], test=Y_data[d:, :],
               train_target=train_target, test_target=test_target, hopt=True)

# Print the error associated with the predictions.
print('Training error:', a['training_error']['rmse_average'])
print('Model error:', a['validation_error']['rmse_average'])

ax = fig.add_subplot(155)
tsne = manifold.TSNE(n_components=nc, init='pca', random_state=0)
fitted = tsne.fit(data[:d, :])
Y_data = fitted.fit_transform(data)
plt.scatter(Y_data[:d, 0], Y_data[:d, 1], c=train_target, cmap=plt.cm.Spectral,
            alpha=0.5)
plt.title("t-SNE")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

print('t-SNE')
a = do_predict(train=Y_data[:d, :], test=Y_data[d:, :],
               train_target=train_target, test_target=test_target, hopt=True)

# Print the error associated with the predictions.
print('Training error:', a['training_error']['rmse_average'])
print('Model error:', a['validation_error']['rmse_average'])

plt.show()
