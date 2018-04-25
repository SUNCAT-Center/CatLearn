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

    def test_learning_curve(self):
        """Test learning curve from DB."""
        hv = Hierarchy(db_name='vec_store.sqlite', file_name='hierarchy')

        # If you want to keep datasize fixed, and vary features.
        featselect_featvar = False
        # If False above. Feature selection for ordinary ridge regression, or
        # just plain vanilla ridge.
        featselect_featconst = True

        # i is the size of featureset.
        i = 10
        lim = i + 2
        if featselect_featvar:
            # Under which interval of feature set in inspection.
            # (Varying of feature size)
            select_limit = [0, 20]
        else:
            # For which fetureset inspection is made. (Varying of data size)
            select_limit = [i - 1, i + 1]
        while i < lim:
            set_size, p_error, result, PC = hierarchy(
                hv, 243, 5, 45, new_data=True, ridge=True, scale=True,
                globalscale=True, normalization=True,
                featselect_featvar=featselect_featvar,
                featselect_featconst=featselect_featconst,
                select_limit=select_limit,
                feat_sub=i)
            if not (set_size and p_error) == [] and not featselect_featvar:
                for data in result:
                    print('data size:', data[0], 'prediction error:',
                          data[1], 'Omega:', data[5],
                          'Euclidean length:', data[2],
                          'Pearson correlation:', data[3])
                i += 1
            elif (set_size and p_error) == [] and not featselect_featvar:
                print("No subset {}".format(i))
                i += 1
                lim += 1
            if featselect_featvar:
                # Don't want to make four subpl for varying of features.
                i += 4
            select_limit = [i - 1, i + 1]

    def test_frequency(self):
        hv = Hierarchy(db_name='vec_store.sqlite', file_name='hierarchy')
        # Produce frequency plots between the lower and upp bound.
        for i in range(20, 22):

            select_limit = [i - 1, i + 1]
            data1 = np.empty(1,)
            data2 = np.empty(1,)
            hit1, hit2 = 0, 0
            for k in range(1, 4):
                selected_features1 = feature_frequency(
                    hv, 243, 3, 8, new_data=True, ridge=True, scale=True,
                    globalscale=True, normalization=True,
                    featselect_featvar=False, featselect_featconst=True,
                    select_limit=select_limit, feat_sub=i)
                selected_features2 = feature_frequency(
                    hv, 243, 3, 8, smallest=True, new_data=False, ridge=True,
                    scale=True, globalscale=True, normalization=True,
                    featselect_featvar=False, featselect_featconst=True,
                    select_limit=select_limit, feat_sub=i)
                if bool(selected_features1):
                    hit1 += 1
                if bool(selected_features2):
                    hit2 += 1
                if bool(selected_features1) and bool(selected_features2):
                    data1 = np.concatenate(
                        (data1, (
                            list(selected_features1.items())[0])[1][0][:]),
                        axis=0)
                    data2 = np.concatenate(
                        (data2, (
                            list(selected_features2.items())[0])[1][0][:]),
                        axis=0)
            data1 = np.delete(data1, 0)
            data2 = np.delete(data2, 0)

            data_all = np.concatenate((data1, data2), axis=0)
            if len(data_all) > 0:
                bins = np.arange(min(data_all) - 2, max(data_all) + 2, 0.5)
                hist1 = np.histogram(data1, bins=bins)
                hist2 = np.histogram(data2, bins=bins)
                r1_hist1 = np.delete(hist1[0], np.where(hist1[0] == 0))
                r1_hist1 = np.divide(
                    r1_hist1.astype('float'), len(data1)) * 100
                r2_hist1 = np.delete(
                    np.delete(hist1[1], np.where(hist1[0] == 0)), -1)

                r1_hist2 = np.delete(hist2[0], np.where(hist2[0] == 0))
                r1_hist2 = np.divide(
                    r1_hist2.astype('float'), len(data2)) * 100
                r2_hist2 = np.delete(
                    np.delete(hist2[1], np.where(hist2[0] == 0)), -1)

                if np.shape(r1_hist2)[0] > np.shape(r1_hist1)[0]:
                    dif = np.shape(r1_hist2)[0] - np.shape(r1_hist1)[0]
                    r1_hist1 = np.concatenate(
                        (r1_hist1, np.zeros(dif)), axis=0)
                    r2_hist1 = np.concatenate(
                        (r2_hist1, np.zeros(dif)), axis=0)
                elif np.shape(r1_hist1)[0] > np.shape(r1_hist2)[0]:
                    dif = np.shape(r1_hist1)[0] - np.shape(r1_hist2)[0]
                    r1_hist2 = np.concatenate(
                        (r1_hist2, np.zeros(dif)), axis=0)
                    r2_hist2 = np.concatenate(
                        (r2_hist2, np.zeros(dif)), axis=0)

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
        train_size, test_size = 45, 5
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
