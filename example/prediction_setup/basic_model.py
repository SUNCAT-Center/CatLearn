"""Basic test for the ML model."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from atoml.cross_validation import HierarchyValidation
from atoml.feature_preprocess import standardize
from atoml.predict import GaussianProcess

# Set numpy print.
np.set_string_function(lambda x:  ', '.join(map(lambda y: '%.3f' % y, x)),
                       repr=False)

# Set some parameters.
plot = True
ds = 500

# Define the hierarchy cv class method.
hv = HierarchyValidation(db_name='../../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='split')
# Split the data into subsets.
hv.split_index(min_split=ds, max_split=ds*2)
# Load data back in from save file.
ind = hv.load_split()

# Split out the various data.
train_data = np.array(hv._compile_split(ind['1_1'])[:, 149:203], np.float64)
train_target = np.array(hv._compile_split(ind['1_1'])[:, -1:], np.float64)
test_data = np.array(hv._compile_split(ind['1_2'])[:, 149:203], np.float64)
test_target = np.array(hv._compile_split(ind['1_2'])[:, -1:], np.float64)

# Scale and shape the data.
std = standardize(train_matrix=train_data, test_matrix=test_data)
train_data, test_data = std['train'], std['test']
train_target = train_target.reshape(len(train_target), )
test_target = test_target.reshape(len(test_target), )


def do_predict(train, test, train_target, test_target, kdict, hopt=False,
               reg=0.001):
    """Function to make predictions."""
    gp = GaussianProcess(train_fp=train, train_target=train_target,
                         kernel_dict=kdict, regularization=reg,
                         optimize_hyperparameters=hopt)

    pred = gp.get_predictions(test_fp=test,
                              test_target=test_target,
                              get_validation_error=True,
                              get_training_error=True)
    return pred, gp


# Start making predictions with linear kernel.
print('\nLinear Kernel\n')
kdict = {'k1': {'type': 'linear', 'const': 10., 'scaling': 1.}}
lopt = do_predict(train=train_data, test=test_data, train_target=train_target,
                  test_target=test_target, kdict=kdict, hopt=True)

lr = lopt[1].regularization
lc = lopt[1].kernel_dict['k1']['const']
ls = lopt[1].kernel_dict['k1']['scaling']
print('regularization is {0:.3f}, \
constant is {1:.3f} and scaling is {2:.3f}'.format(lr, lc, ls))

# Print the error associated with the predictions.
print('Training error: {0:.3f}'.format(
    lopt[0]['training_error']['rmse_average']))
print('Model error: {0:.3f}'.format(
    lopt[0]['validation_error']['rmse_average']))

# Compare predictions with gaussian kernel.
print('\nGaussian Kernel\n')
kdict = {'k1': {'type': 'gaussian', 'width': 10., 'scaling': 1.}}
gopt = do_predict(train=train_data, test=test_data, train_target=train_target,
                  test_target=test_target, kdict=kdict, hopt=True)

gr = gopt[1].regularization
gc = np.array(gopt[1].kernel_dict['k1']['width'])
gs = gopt[1].kernel_dict['k1']['scaling']
print('regularization is {0:.3f}, \
width is [{1}] and scaling is {2:.3f}'.format(gr, gc, gs))

# Print the error associated with the predictions.
print('Training error: {0:.3f}'.format(
    gopt[0]['training_error']['rmse_average']))
print('Model error: {0:.3f}'.format(
    gopt[0]['validation_error']['rmse_average']))

# Compare predictions with mixed linear/gaussian kernel.
print('\nLinear Gaussian Mixing\n')
kdict = {'k1': {'type': 'linear', 'const': lc, 'scaling': ls},
         'k2': {'type': 'gaussian', 'width': gc, 'scaling': gs}}
mopt = do_predict(train=train_data, test=test_data, train_target=train_target,
                  test_target=test_target, kdict=kdict, hopt=True, reg=lr)

mr = mopt[1].regularization
mc = mopt[1].kernel_dict['k1']['const']
ms1 = mopt[1].kernel_dict['k1']['scaling']
mw = np.array(mopt[1].kernel_dict['k2']['width'])
ms2 = mopt[1].kernel_dict['k2']['scaling']
print('regularization is {0:.3f}, \
constant is {1:.3f} and scaling is {2:.3f}'.format(mr, mc, ms1))
print('regularization is {0:.3f}, \
width is [{1}] and scaling is {2:.3f}'.format(mr, mw, ms2))

# Print the error associated with the predictions.
print('Training error: {0:.3f}'.format(
    mopt[0]['training_error']['rmse_average']))
print('Model error: {0:.3f}'.format(
    mopt[0]['validation_error']['rmse_average']))

if plot:
    fig = plt.figure(figsize=(15, 8))
    sns.axes_style('dark')
    sns.set_style('ticks')

    # Setup pandas dataframes.
    index = [i for i in range(len(test_data))]
    lopt[0]['actual'] = test_target
    lopt_df = pd.DataFrame(data=lopt[0], index=index)
    gopt[0]['actual'] = test_target
    gopt_df = pd.DataFrame(data=gopt[0], index=index)
    mopt[0]['actual'] = test_target
    mopt_df = pd.DataFrame(data=mopt[0], index=index)

    ax = fig.add_subplot(131)
    sns.regplot(x='actual', y='prediction', data=lopt_df)
    plt.title('Linear Validation RMSE: {0:.3f}'.format(
        lopt[0]['validation_error']['rmse_average']))

    ax = fig.add_subplot(132)
    sns.regplot(x='actual', y='prediction', data=gopt_df)
    plt.title('Gaussian Validation RMSE: {0:.3f}'.format(
        gopt[0]['validation_error']['rmse_average']))

    ax = fig.add_subplot(133)
    sns.regplot(x='actual', y='prediction', data=mopt_df)
    plt.title('Mixed Validation RMSE: {0:.3f}'.format(
        mopt[0]['validation_error']['rmse_average']))

    plt.show()
