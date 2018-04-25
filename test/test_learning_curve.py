"""Run the hierarchy with feature selection."""
import numpy as np
import unittest
import os

from catlearn.cross_validation import Hierarchy
from catlearn.learning_curve import hierarchy, feature_frequency
from catlearn.utilities import DescriptorDatabase
from catlearn.regression import GaussianProcess
from catlearn.utilities.utilities import LearningCurve

wkdir = os.getcwd()


class TestCurve(unittest.TestCase):
    """Test out the learning curve functions."""

    def get_data(self):
        """Simple function to pull some training and test data."""
        # Attach the database.
        wkdir = os.getcwd()
        dd = DescriptorDatabase(db_name='{}/vec_store.sqlite'.format(wkdir),
                                table='FingerVector')

        # Pull the features and targets from the database.
        names = dd.get_column_names()
        features, targets = names[1:-1], names[-1:]
        feature_data = dd.query_db(names=features)
        target_data = np.reshape(dd.query_db(names=targets),
                                 (np.shape(feature_data)[0], ))
        train_size, test_size = 20, 5
        n_features = 20
        # Split the data into so test and training sets.
        train_features = feature_data[:train_size, :n_features]
        train_targets = target_data[:train_size]
        test_features = feature_data[test_size:, :n_features]
        test_targets = target_data[test_size:]

        return train_features, train_targets, test_features, test_targets

    def prediction(self, train_features, train_targets,
                   test_features, test_targets):
        """Ridge regression predictions."""
        # Test ridge regression predictions.
        sigma = 1.
        kdict = {'gk': {'type': 'gaussian', 'width': sigma,
                        'dimension': 'single'}
                 }
        regularization = 0.001
        gp = GaussianProcess(train_fp=train_features,
                             train_target=train_targets,
                             kernel_dict=kdict,
                             regularization=regularization,
                             optimize_hyperparameters=False,
                             scale_data=True)
        output = gp.predict(test_fp=test_features,
                            test_target=test_targets,
                            get_validation_error=True,
                            get_training_error=True,
                            uncertainty=True)
        r = [np.shape(train_features)[0],
             output['validation_error']['rmse_average'],
             output['validation_error']['absolute_average'],
             np.mean(output['uncertainty'])
             ]
        return r

    def test_simple_learn(self):
        [train_features, train_targets,
         test_features, test_targets] = self.get_data()
        ge = LearningCurve(nprocs=1)
        step = 5
        min_data = 10
        output = ge.learning_curve(self.prediction,
                                   train_features, train_targets,
                                   test_features, test_targets,
                                   step=step, min_data=min_data)
        print(np.shape(output))


if __name__ == '__main__':
    unittest.main()
