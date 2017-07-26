"""A simple example for the hierarchy CV function."""
import numpy as np
from atoml.cross_validation import HierarchyValidation
from atoml.ridge_regression import RidgeRegression
from atoml.feature_preprocess import standardize
from atoml.predict import target_standardize

# Define the hierarchey cv class method.
hv = HierarchyValidation(db_name='../../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='hierarchy')
# Split the data into subsets.
hv.split_index(min_split=50, max_split=2000)
# Load data back in from save file.
ind = hv.load_split()


def predict(train_features, train_targets, test_features, test_targets,
            features):
    """Function to perform the prediction."""
    data = {}

    # Set how many features to include in the model.
    train_features = train_features[:, :features]
    test_features = test_features[:, :features]

    std = standardize(train_matrix=train_features, test_matrix=test_features)
    train_features = std['train']
    test_features = std['test']

    ts = target_standardize(train_targets)
    train_targets = ts['target']

    # Set up the ridge regression function.
    rr = RidgeRegression(W2=None, Vh=None, cv='loocv')
    b = rr.find_optimal_regularization(X=train_features, Y=train_targets)
    coef = rr.RR(X=train_features, Y=train_targets, omega2=b)[0]

    # Test the model.
    sumd = 0.
    for tf, tt in zip(test_features, test_targets):
        p = (np.dot(coef, tf) * ts['std']) + ts['mean']
        sumd += (p - tt) ** 2
    error = (sumd / len(test_features)) ** 0.5

    data['result'] = len(test_targets), error

    return data


# Make the predictions for each subset.
res = hv.split_predict(index_split=ind, predict=predict, features=200)
# Print out the errors.
for i in res:
    print(i[0], i[1])
