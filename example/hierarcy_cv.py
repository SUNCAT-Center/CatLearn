"""A simple example for the hierarchy CV function."""
import numpy as np
from atoml.cross_validation import HierarchyValidation
from atoml.ridge_regression import find_optimal_regularization, RR

# Define the hierarchey cv class method.
hv = HierarchyValidation(db_name='train_db.sqlite', table='FingerVector',
                         file_name='split')
# Split the data into subsets.
hv.split_index(min_split=50, max_split=2000)
# Load data back in from save file.
ind = hv.load_split()


def predict(train_features, train_targets, test_features, test_targets):
    """Function to perform the prediction."""
    data = {}
    b = find_optimal_regularization(X=train_features, Y=train_targets, p=0,
                                    Ns=100)
    coef = RR(X=train_features, Y=train_targets, p=0, omega2=b, W2=None,
              Vh=None)[0]

    # Test the model.
    sumd = 0.
    for tf, tt in zip(test_features, test_targets):
        sumd += (np.dot(coef, tf) - tt) ** 2
    error = (sumd / len(test_features)) ** 0.5

    data['result'] = len(test_targets), error

    return data


# Make the predictions for each subset.
res = hv.split_predict(index_split=ind, predict=predict)
# Print out the errors.
for i in res:
    print(i[0], i[1])
