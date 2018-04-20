"""Test hierarchy CV functions."""
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import unittest

from catlearn.cross_validation import Hierarchy, k_fold
from catlearn.cross_validation.k_fold_cv import write_split, read_split
from catlearn.regression import RidgeRegression
from common import get_data

wkdir = os.getcwd()


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


class TestValidation(unittest.TestCase):
    """Test out the hierarchy cv."""

    def test_hierarchy(self):
        """Function to test the hierarchy with ridge regression predictions."""
        # Define the hierarchy cv class method.
        train_features, train_targets, test_features, test_targets = get_data()

        hv = Hierarchy(
            db_name='{}/test.sqlite'.format(wkdir), file_name='hierarchy')
        hv.todb(features=train_features, targets=train_targets)
        # Split the data into subsets.
        split = hv.split_index(min_split=5, max_split=25)
        self.assertEqual(len(split), 6)
        # Load data back in from save file.
        ind = hv.load_split()
        self.assertEqual(len(ind), len(split))

        # Make the predictions for each subset.
        pred = hv.split_predict(index_split=ind, predict=predict)
        self.assertTrue(len(pred[0]) == 14 and len(pred[1]) == 14)

    def test_kfold(self):
        """Test some cross-validation."""
        features, targets, _, _ = get_data()
        f, t = k_fold(features, nsplit=5, targets=targets)
        self.assertTrue(len(f) == 5 and len(t) == 5)
        for s in f:
            self.assertEqual(np.shape(s), (9, 100))
        f, t = k_fold(features, nsplit=4, targets=targets, fix_size=5)
        self.assertTrue(len(f) == 4 and len(t) == 4)
        for s in f:
            self.assertEqual(np.shape(s), (5, 100))

        write_split(features=f, targets=t, fname='cvsave', fformat='pickle')
        f1, t1 = read_split(fname='cvsave', fformat='pickle')
        self.assertEqual(len(f1), len(f))
        self.assertEqual(len(t1), len(t))

        write_split(features=f, targets=t, fname='cvsave', fformat='json')
        f1, t1 = read_split(fname='cvsave', fformat='pickle')
        self.assertEqual(len(f1), len(f))
        self.assertEqual(len(t1), len(t))


if __name__ == '__main__':
    unittest.main()
