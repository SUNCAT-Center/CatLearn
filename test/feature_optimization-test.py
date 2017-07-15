"""Script to test feature space optimization functions."""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from atoml.database_functions import DescriptorDatabase
from atoml.feature_engineering import (single_transform, get_order_2,
                                       get_div_order_2, get_order_2ab,
                                       get_ablog)
from atoml.feature_elimination import FeatureScreening

# Attach the database.
dd = DescriptorDatabase(db_name='fpv_store.sqlite', table='FingerVector')

# Pull the features and targets from the database.
names = dd.get_column_names()
features, targets = names[1:-1], names[-1:]
feature_data = dd.query_db(names=features)
target_data = np.reshape(dd.query_db(names=targets),
                         (np.shape(feature_data)[0], ))

# Split the data into so test and training sets.
train_features, train_targets = feature_data[:35, :], target_data[:35]
test_features, test_targets = feature_data[35:, :], target_data[35:]
d, f = np.shape(train_features)

# Perform feature engineering.
extend = single_transform(train_features)
assert np.shape(extend) == (d, f * 3)

extend = get_order_2(train_features)
assert np.shape(extend) == (d, 29403)

extend = get_div_order_2(train_features)
assert np.shape(extend) == (d, 58564)

extend = get_order_2ab(train_features, a=2, b=4)
assert np.shape(extend) == (d, 29403)

extend = get_ablog(train_features, a=2, b=4)
assert np.shape(extend) == (d, 29403)

# Get descriptor correlation
corr = ['pearson', 'spearman', 'kendall']
for c in corr:
    screen = FeatureScreening(correlation=c, iterative=False)
    features = screen.eliminate_features(target=train_targets,
                                         train_features=train_features,
                                         test_features=test_features,
                                         size=d, step=None, order=None)
    assert np.shape(features[0])[1] == d and np.shape(features[1])[1] == d

    screen = FeatureScreening(correlation=c, iterative=True,
                              regression='lasso')
    features = screen.eliminate_features(target=train_targets,
                                         train_features=train_features,
                                         test_features=test_features,
                                         size=d, step=2, order=None)
    assert np.shape(features[0])[1] == d and np.shape(features[1])[1] == d
