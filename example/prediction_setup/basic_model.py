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

# Decide whether to remove output and print graph.
cleanup = True
plot = False
ds = 500

# Define the hierarchey cv class method.
hv = HierarchyValidation(db_name='../../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='split')
# Split the data into subsets.
hv.split_index(min_split=ds, max_split=ds*2)
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


def do_predict(train, test, train_target, test_target, hopt=False):
    """Function to make predictions."""
    pred = gp.get_predictions(train_fp=train,
                              test_fp=test,
                              train_target=train_target,
                              test_target=test_target,
                              get_validation_error=True,
                              get_training_error=True,
                              optimize_hyperparameters=hopt)

    if plot:
        pred['actual'] = test_target
        index = [i for i in range(len(test_data))]
        df = pd.DataFrame(data=pred, index=index)
        with sns.axes_style("white"):
            sns.regplot(x='actual', y='prediction', data=df)
        plt.title('Validation RMSE: {0:.3f}'.format(
            pred['validation_rmse']['average']))
        plt.show()

    return pred


# Set up the prediction routine.
kdict = {'k1': {'type': 'gaussian', 'width': 10.}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)

print('Original parameters')
a = do_predict(train=train_data, test=test_data, train_target=train_target,
               test_target=test_target, hopt=False)

# Print the error associated with the predictions.
print('Training error:', a['training_rmse']['average'])
print('Model error:', a['validation_rmse']['average'])

# Try with hyperparameter optimization.
kdict = {'k1': {'type': 'gaussian', 'width': 10.}}
gp = GaussianProcess(kernel_dict=kdict, regularization=0.001)

print('Optimized parameters')
a = do_predict(train=train_data, test=test_data, train_target=train_target,
               test_target=test_target, hopt=True)

# Print the error associated with the predictions.
print('Training error:', a['training_rmse']['average'])
print('Model error:', a['validation_rmse']['average'])
