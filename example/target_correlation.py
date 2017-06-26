"""Plot correlation between features."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

from atoml.cross_validation import HierarchyValidation
from atoml.feature_preprocess import standardize
from atoml.predict import GaussianProcess

triangle = True  # Plot only the bottom half of the matrix
divergent = False  # Plot with a diverging color palette.
absolute = True  # Take the absolute values for the correlation.
clean = True  # Clean zero-varience features
expand = False

correlation = 'pearson'

mask = None

# Define the hierarchey cv class method.
hv = HierarchyValidation(db_name='../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='split')
# Split the data into subsets.
hv.split_index(min_split=200, max_split=2000)
# Load data back in from save file.
ind = hv.load_split()

train_data = np.array(hv._compile_split(ind['1_1'])[:, 1:-1], np.float64)
train_target = np.array(hv._compile_split(ind['1_1'])[:, -1:], np.float64)

test_data = np.array(hv._compile_split(ind['1_2'])[:, 1:-1], np.float64)
test_target = np.array(hv._compile_split(ind['1_2'])[:, -1:], np.float64)

# d, f = np.shape(train_data)

std = standardize(train_matrix=train_data, test_matrix=test_data)
train_data, test_data = std['train'], std['test']

train_target = train_target.reshape(len(train_target), )
test_target = test_target.reshape(len(test_target), )

# Compute the correlation matrix
regr = LassoCV(fit_intercept=True, normalize=True, n_alphas=100, eps=1e-3,
               cv=None)
model = regr.fit(X=train_data, y=train_target)
coeff_lasso = model.coef_

regr = RidgeCV(fit_intercept=True, normalize=True)
model = regr.fit(X=train_data, y=train_target)
coeff_ridge = regr.coef_

l1_list = [0.1, 0.25, 0.5, 0.75, 0.9]
regr = ElasticNetCV(fit_intercept=True, normalize=True,
                    l1_ratio=0.5)
model = regr.fit(X=train_data, y=train_target)
coeff_elast = regr.coef_

corr = []
for c in train_data.T:
    if correlation is 'pearson':
        corr.append(pearsonr(c, train_target)[0])
    elif correlation is 'spearman':
        corr.append(spearmanr(c, train_target)[0])
    elif correlation is 'kendall':
        corr.append(kendalltau(c, train_target)[0])
ind = list(range(len(corr)))

print('Lasso none zero coefs:', (coeff_lasso != 0.).sum())
print('Elastic none zero coefs:', (coeff_elast != 0.).sum())

nz_lasso = (coeff_lasso != 0.).sum()
index = list(range(len(coeff_lasso)))
sort_list_lasso = [list(i) for i in zip(*sorted(zip(np.abs(coeff_lasso),
                                                    index, corr),
                                                key=lambda x: x[0],
                                                reverse=True))]

sort_list_ridge = [list(i) for i in zip(*sorted(zip(np.abs(coeff_ridge),
                                                    index, corr),
                                                key=lambda x: x[0],
                                                reverse=True))]

sort_list_elast = [list(i) for i in zip(*sorted(zip(np.abs(coeff_elast),
                                                    index, corr),
                                                key=lambda x: x[0],
                                                reverse=True))]

delf = sort_list_lasso[1][nz_lasso:]
traind = np.delete(train_data, delf, axis=1)
testd = np.delete(test_data, delf, axis=1)
# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': 5.}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)
# Do the predictions.
lasso_pred = gp.get_predictions(train_fp=traind,
                                test_fp=testd,
                                train_target=train_target,
                                test_target=test_target,
                                get_validation_error=True,
                                get_training_error=True,
                                optimize_hyperparameters=True)
# Print the error associated with the predictions.
print('Training error:', lasso_pred['training_rmse']['average'])
print('Model error:', lasso_pred['validation_rmse']['average'])
print('width:', lasso_pred['optimized_kernels']['k1']['width'], 'reg:',
      lasso_pred['optimized_regularization'])

delf = sort_list_ridge[1][nz_lasso:]
traind = np.delete(train_data, delf, axis=1)
testd = np.delete(test_data, delf, axis=1)
# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': 5.}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)
# Do the predictions.
ridge_pred = gp.get_predictions(train_fp=traind,
                                test_fp=testd,
                                train_target=train_target,
                                test_target=test_target,
                                get_validation_error=True,
                                get_training_error=True,
                                optimize_hyperparameters=True)
# Print the error associated with the predictions.
print('Training error:', ridge_pred['training_rmse']['average'])
print('Model error:', ridge_pred['validation_rmse']['average'])
print('width:', ridge_pred['optimized_kernels']['k1']['width'], 'reg:',
      ridge_pred['optimized_regularization'])

delf = sort_list_elast[1][nz_lasso:]
traind = np.delete(train_data, delf, axis=1)
testd = np.delete(test_data, delf, axis=1)
# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': 5.}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)
# Do the predictions.
elast_pred = gp.get_predictions(train_fp=traind,
                                test_fp=testd,
                                train_target=train_target,
                                test_target=test_target,
                                get_validation_error=True,
                                get_training_error=True,
                                optimize_hyperparameters=True)
# Print the error associated with the predictions.
print('Training error:', elast_pred['training_rmse']['average'])
print('Model error:', elast_pred['validation_rmse']['average'])
print('width:', elast_pred['optimized_kernels']['k1']['width'], 'reg:',
      elast_pred['optimized_regularization'])

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(141)
ax.plot(ind, np.abs(corr), '-', color='black', lw=1)
ax.plot(sort_list_lasso[1][:nz_lasso],
        np.abs(sort_list_lasso[2][:nz_lasso]), 'o', color='blue',
        alpha=0.45)
ax.plot(sort_list_ridge[1][:nz_lasso],
        np.abs(sort_list_ridge[2][:nz_lasso]), 'o', color='red',
        alpha=0.45)
ax.plot(sort_list_elast[1][:nz_lasso],
        np.abs(sort_list_elast[2][:nz_lasso]), 'o', color='green',
        alpha=0.45)
ax.axis([0, len(ind), 0, 1])
plt.title(correlation)
plt.xlabel('Feature No.')
plt.ylabel('Correlation')

ax = fig.add_subplot(142)
ax.plot(test_target, lasso_pred['prediction'], 'o', color='blue', alpha=0.45)
plt.title('Validation RMSE: {0:.3f}'.format(
    lasso_pred['validation_rmse']['average']))
plt.xlabel('Actual (eV)')
plt.ylabel('Prediction (eV)')

ax = fig.add_subplot(143)
ax.plot(test_target, ridge_pred['prediction'], 'o', color='red', alpha=0.45)
plt.title('Validation RMSE: {0:.3f}'.format(
    ridge_pred['validation_rmse']['average']))
plt.xlabel('Actual (eV)')
plt.ylabel('Prediction (eV)')

ax = fig.add_subplot(144)
ax.plot(test_target, elast_pred['prediction'], 'o', color='green', alpha=0.45)
plt.title('Validation RMSE: {0:.3f}'.format(
    elast_pred['validation_rmse']['average']))
plt.xlabel('Actual (eV)')
plt.ylabel('Prediction (eV)')

plt.show()
