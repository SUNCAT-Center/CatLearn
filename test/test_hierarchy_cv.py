"""Test hierarchy CV functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

from atoml.cross_validation import Hierarchy
from atoml.regression import RidgeRegression
from common import get_data


def predict(train_features, train_targets, test_features, test_targets):
    """Function to perform the prediction."""
    data = {}

    # Set up the ridge regression function.
    rr = RidgeRegression(W2=None, Vh=None, cv='loocv')
    b = rr.find_optimal_regularization(X=train_features, Y=train_targets)
    coef = rr.RR(X=train_features, Y=train_targets, omega2=b)[0]

    # Test the model.
    sumd = 0.
    err = []
    for tf, tt in zip(test_features, test_targets):
        p = np.dot(coef, tf)
        sumd += (p - tt) ** 2
        e = ((p - tt) ** 2) ** 0.5
        err.append(e)
    error = (sumd / len(test_features)) ** 0.5

    data['result'] = error
    data['size'] = len(train_targets)

    return data


def hierarchy_test():
    """Function to test the hierarchy with ridge regression predictions."""
    # Define the hierarchy cv class method.
    train_features, train_targets, test_features, test_targets = get_data()
    hv = Hierarchy(db_name='test.sqlite', table='FingerVector',
                   file_name='hierarchy')
    hv.todb(features=train_features, targets=train_targets)
    # Split the data into subsets.
    hv.split_index(min_split=5, max_split=None)
    # Load data back in from save file.
    ind = hv.load_split()

    # Make the predictions for each subset.
    hv.split_predict(index_split=ind, predict=predict)

    os.remove('hierarchy.pickle')
    os.remove('test.sqlite')


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    hierarchy_test()

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
