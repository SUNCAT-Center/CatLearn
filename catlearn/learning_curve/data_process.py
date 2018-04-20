"""Processing of data for HierarchyValidation."""
from __future__ import division

import numpy as np
from scipy.stats import pearsonr

from catlearn.preprocess.scaling import (
    standardize, normalize, target_standardize, target_normalize)


class data_process(object):
    """Class to glue different function used for HierarchyValidation.

    This class pick up data from HierarchyValidation. The data is then modified
    if requested with "feature_preprocess", and "predict". The data is then
    fitted with regression model for example with "ridge_regression". The error
    of the fit is then measured.
    """

    def __init__(self, features, min_split, max_split,
                 scale=True, normalization=True, ridge=True, loocv=True,
                 batchfarm=False):
        """Data_process setup.

        Parameters
        ----------
        features : int
            Number of features used for regression.
        min_split : int
            Number of datasplit in the smallest sub-set.
        max_split : int
            Number of datasplit in the largest sub-set.
        scale : string
            If the data are supposed to be scaled or not.
        normalization : string
            If scaled, normalized or standardized. Normalized is default.
        ridge : string
            Ridge regulazer is deafult. If False, lasso is used.
        """
        self.features = features
        self.min_split = min_split
        self.max_split = max_split
        self.scale = scale
        self.normalization = normalization
        self.ridge = ridge
        self.loocv = loocv
        self.batchfarm = batchfarm

    def scaling_data(self, train_features, train_targets, test_features,
                     s_tar, m_tar, s_feat, m_feat):
        """Scaling the data if requested.

        Parameters
        ----------
        train_feature : array
            Independent data used to train model.
        train_targets : array
            Dependent data used to train model.
        test_features : array
            Independent data used to test the model.
        s_tar : array
            Standard devation or (max-min), for the dependent train_targets.
        m_tar : array
            Mean for the dependent train_targets.
        s_feat : array
            Standard devation or (max-min), for the independent train_features.
        m_feat : array
            Mean for the independent train_features.
        """
        train_features = train_features[:, :self.features]
        test_features = test_features[:, :self.features]

        if self.scale:
            if self.normalization:
                # Normalize
                if s_tar is None or m_tar is None:
                    data = target_normalize(target=train_targets)
                    s_tar, m_tar, train_targets = (data['dif'], data['mean'],
                                                   data['target'])

                norm = normalize(train_matrix=train_features,
                                 test_matrix=test_features,
                                 dif=s_feat, mean=m_feat)
                train_features, test_features, s_feat, m_feat = (norm['train'],
                                                                 norm['test'],
                                                                 norm['dif'],
                                                                 norm['mean'])
            else:
                # Standardization
                if s_tar is None or m_tar is None:
                    data = target_standardize(target=train_targets)
                    s_tar, m_tar, train_targets = (data['std'], data['mean'],
                                                   data['target'])

                std = standardize(train_matrix=train_features,
                                  test_matrix=test_features,
                                  std=s_feat, mean=m_feat)
                train_features, test_features, s_feat, m_feat = (std['train'],
                                                                 std['test'],
                                                                 std['std'],
                                                                 std['mean'])
        return (s_tar, m_tar, s_feat, m_feat, train_targets, train_features,
                test_features)

    def globalscaling(self, globalscaledata, train_features):
        """All sub-groups of traindata are scaled same.

        Parameters
        ----------
        globalscaledata : string
            The data will be scaled globally if requested.
        """
        g_data = globalscaledata[:, :self.features]

        if self.normalization:

            norm = normalize(train_matrix=g_data)
            s_feat, m_feat = norm['dif'], norm['mean']

            norm = normalize(train_matrix=train_features,
                             test_matrix=None,
                             dif=s_feat, mean=m_feat)
            train_features = (norm['train'])

        else:
            std = standardize(train_matrix=g_data)
            s_feat, m_feat = std['std'], std['mean']
        return s_feat, m_feat, train_features

    def prediction_error(self, test_features, test_targets, coef, s_tar,
                         m_tar):
        """Calculate the error of the prediction with the model.

        Parameters
        ----------
        test_features : array
            Independet data for testing the model.
        test_targets : array
            Dependent data to test the model.
        coef : array
            The coeffieiceints which makes up the model.
        s_tar : string
            Standard devation or (max-min), for the dependent train_targets.
        m_tar : array
            Mean for the dependent train_targets.
        """
        data = {}
        sumd = 0.
        p_corr = []

        if self.scale:
            for tf, tt, in zip(test_features, test_targets):
                sumd += (np.dot(coef, tf) * s_tar + m_tar - tt)**2
                p_corr.append(np.dot(coef, tf))
        else:
            for tf, tt, in zip(test_features, test_targets):
                sumd += (np.dot(coef, tf) - tt) ** 2
                p_corr.append(np.dot(coef, tf))

        error = (sumd / len(test_features)) ** 0.5

        ecludian_length = np.sqrt(np.dot(coef, coef))

        data['result'] = (len(test_features), error, ecludian_length,
                          pearsonr(p_corr, test_targets)[0])

        return data

    def get_statistic(self, data_size, p_error):
        """Generate statistics for predicition.

        Parameters
        ----------
        data_size : list
            Data_size for where the prediction were made.
        p_error : list
            Error for where the prediction were made.
        """
        p_error_list, data_size_list = [], []
        p_error_mean, data_size_mean, k = [], [], 0
        for size, error in zip(data_size, p_error):
            if size > k + 1:
                k = size
                p_error_list.append(p_error_mean)
                data_size_list.append(data_size_mean)
                p_error_mean = [error]
                data_size_mean = [size]
            else:
                p_error_mean.append(error)
                data_size_mean.append(size)
        p_error_list.append(p_error_mean)
        data_size_list.append(data_size_mean)
        del p_error_list[0], data_size_list[0]
        p_error_mean_list, data_size_mean_list, corrected_std \
            = self.average_nested(Y=p_error_list, X=data_size_list)
        return p_error_mean_list, data_size_mean_list, corrected_std

    def average_nested(self, Y, X):
        """Calculate statistics for predicition.

        Parameters
        ----------
        data_size : list
            Data_size for where the prediction were made.
        p_error : list
            Error for where the prediction were made.
        """
        corrected_std, Y_mean, X_mean = [], [], []
        for listx, listy in zip(X, Y):
            X_mean.append(sum(listx) / len(listx))
            Y_mean.append(sum(listy) / len(listy))
            summa = 0
            for y in listy:
                summa += (y - Y_mean[-1])**2

            corrected_std.append((summa / (len(listy) - 1))**0.5)

        return Y_mean, X_mean, corrected_std
