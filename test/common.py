"""Common functions for the test suite."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os

from atoml.utilities import DescriptorDatabase

wkdir = os.getcwd()

train_size, test_size = 45, 5


def get_data():
    """Simple function to pull some training and test data."""
    # Attach the database.
    dd = DescriptorDatabase(db_name='{}/fpv_store.sqlite'.format(wkdir),
                            table='FingerVector')

    # Pull the features and targets from the database.
    names = dd.get_column_names()
    features, targets = names[1:-1], names[-1:]
    feature_data = dd.query_db(names=features)
    target_data = np.reshape(dd.query_db(names=targets),
                             (np.shape(feature_data)[0], ))

    # Split the data into so test and training sets.
    train_features = feature_data[:train_size, :100]
    train_targets = target_data[:train_size]
    test_features = feature_data[test_size:, :100]
    test_targets = target_data[test_size:]

    return train_features, train_targets, test_features, test_targets
