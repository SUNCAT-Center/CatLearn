"""Plot correlation between features."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

from atoml.cross_validation import HierarchyValidation
from atoml.feature_preprocess import standardize
from atoml.predict import GaussianProcess, get_error

correlation = 'pearson'

# Define the hierarchy cv class method.
hv = HierarchyValidation(db_name='../../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='split')
# Split the data into subsets.
hv.split_index(min_split=50, max_split=2000)
# Load data back in from save file.
ind = hv.load_split()

# Split out the various data.
train_data = np.array(hv._compile_split(ind['1_1'])[:, 1:-1], np.float64)
train_target = np.array(hv._compile_split(ind['1_1'])[:, -1:], np.float64)
test_data = np.array(hv._compile_split(ind['1_2'])[:, 1:-1], np.float64)
test_target = np.array(hv._compile_split(ind['1_2'])[:, -1:], np.float64)

# Scale and shape the data.
std = standardize(train_matrix=train_data, test_matrix=test_data)
train_data, test_data = std['train'], std['test']
train_target = train_target.reshape(len(train_target), )
test_target = test_target.reshape(len(test_target), )

# Compute the correlation matrix
regr = LassoCV(fit_intercept=True, normalize=True, n_alphas=100, eps=1e-3,
               cv=None)
model = regr.fit(X=train_data, y=train_target)
lasso_func = regr.predict(test_data)
lasso_err = get_error(lasso_func, test_target)
coeff_lasso = model.coef_

regr = RidgeCV(fit_intercept=True, normalize=True)
model = regr.fit(X=train_data, y=train_target)
ridge_func = regr.predict(test_data)
ridge_err = get_error(ridge_func, test_target)
coeff_ridge = regr.coef_

l1_list = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
regr = ElasticNetCV(fit_intercept=True, normalize=True,
                    l1_ratio=l1_list)
model = regr.fit(X=train_data, y=train_target)
elast_func = regr.predict(test_data)
elast_err = get_error(elast_func, test_target)
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
gp = GaussianProcess(train_fp=traind, train_target=train_target,
                     kernel_dict=kdict, regularization=0.001,
                     optimize_hyperparameters=True)
# Do the predictions.
lasso_pred = gp.get_predictions(test_fp=testd,
                                test_target=test_target,
                                get_validation_error=True,
                                get_training_error=True)
# Print the error associated with the predictions.
print('Training error:', lasso_pred['training_error']['rmse_average'])
print('Model error:', lasso_pred['validation_error']['rmse_average'])
print('width:', gp.kernel_dict['k1']['width'], 'reg:', gp.regularization)

delf = sort_list_ridge[1][nz_lasso:]
traind = np.delete(train_data, delf, axis=1)
testd = np.delete(test_data, delf, axis=1)
# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': 5.}}
gp = GaussianProcess(train_fp=traind, train_target=train_target,
                     kernel_dict=kdict, regularization=0.001,
                     optimize_hyperparameters=True)
# Do the predictions.
ridge_pred = gp.get_predictions(test_fp=testd,
                                test_target=test_target,
                                get_validation_error=True,
                                get_training_error=True)
# Print the error associated with the predictions.
print('Training error:', ridge_pred['training_error']['rmse_average'])
print('Model error:', ridge_pred['validation_error']['rmse_average'])
print('width:', gp.kernel_dict['k1']['width'], 'reg:', gp.regularization)

delf = sort_list_elast[1][nz_lasso:]
traind = np.delete(train_data, delf, axis=1)
testd = np.delete(test_data, delf, axis=1)
# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': 5.}}
gp = GaussianProcess(train_fp=traind, train_target=train_target,
                     kernel_dict=kdict, regularization=0.001,
                     optimize_hyperparameters=True)
# Do the predictions.
elast_pred = gp.get_predictions(test_fp=testd,
                                test_target=test_target,
                                get_validation_error=True,
                                get_training_error=True)
# Print the error associated with the predictions.
print('Training error:', elast_pred['training_error']['rmse_average'])
print('Model error:', elast_pred['validation_error']['rmse_average'])
print('width:', gp.kernel_dict['k1']['width'], 'reg:', gp.regularization)

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
ax.plot(test_target, lasso_pred['prediction'], 'o', color='magenta', alpha=0.2)
ax.plot(test_target, lasso_func, 'o', color='blue', alpha=0.2)
plt.title('Func RMSE: {0:.3f}, GP RMSE: {1:.3f}'.format(
    lasso_err['rmse_average'], lasso_pred['validation_error']['rmse_average']))
plt.xlabel('Actual (eV)')
plt.ylabel('Prediction (eV)')

ax = fig.add_subplot(143)
ax.plot(test_target, ridge_pred['prediction'], 'o', color='magenta', alpha=0.2)
ax.plot(test_target, ridge_func, 'o', color='red', alpha=0.2)
plt.title('Func RMSE: {0:.3f}, GP RMSE: {1:.3f}'.format(
    ridge_err['rmse_average'], ridge_pred['validation_error']['rmse_average']))
plt.xlabel('Actual (eV)')
plt.ylabel('Prediction (eV)')

ax = fig.add_subplot(144)
ax.plot(test_target, elast_pred['prediction'], 'o', color='magenta', alpha=0.2)
ax.plot(test_target, elast_func, 'o', color='green', alpha=0.2)
plt.title('Func RMSE: {0:.3f}, GP RMSE: {1:.3f}'.format(
    elast_err['rmse_average'], elast_pred['validation_error']['rmse_average']))
plt.xlabel('Actual (eV)')
plt.ylabel('Prediction (eV)')

plt.show()
