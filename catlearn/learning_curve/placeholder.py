"""Placeholder for now."""
import numpy as np

from catlearn.regression import RidgeRegression
from .feature_selection import feature_selection


class placeholder(object):
    """Used to make the hierarchey more easy to follow.

    Placeholder for now.
    """

    def __init__(self, PC, index_split, hv, indicies,
                 hier_level, featselect_featvar, featselect_featconst,
                 s_feat, m_feat, feat_sub=15, s_tar=None, m_tar=None,
                 select_limit=None, selected_features=None,
                 glob_feat1=None, glob_tar1=None, new_training=True):
        """Function to extract raw data from the database.

        Parameters
        ----------
        PC : object
           Modieifes data such as scaling.
        index_split : array
           Indexed data.
        hv : object
           HierarchyValidation used to get the data.
        indicies : string
           Indicies for the data used.
        hier_level: int
           Counts the level in the hierarchy.
        featselect_featvar : string
           Use feature selection with variation of features.
        featselect_featconst : string
           Use feature selection varying of the datasize.
        s_tar : array
            Standard devation or (max-min), for the dependent train_targets.
        m_tar : array
            Mean for the dependent train_targets.
        s_feat : array
            Standard devation or (max-min), for the independent train_features.
        m_feat : array
            Mean for the independent train_features.
        feature_selection : string
           Using feature selection with ridge, or plain vanilla ridge.
        select_limit : list
           Upper and lower limit of features used for feature selection.
        selected_features : dict
           Contains all the features used.
        feat_sub : int
           The feature subset.
        glob_feat1 : array
           Independent data with the one of the two largest subsets.
        glob_tar1 : array
           Dependent data with the one of the two largest subsets,
        """
        self.PC = PC
        self.index_split = index_split
        self.indicies = indicies
        self.hier_level = hier_level
        self.featselect_featvar = featselect_featvar
        self.s_tar = s_tar
        self.m_tar = m_tar
        self.s_feat = s_feat
        self.m_feat = m_feat
        self.select_limit = select_limit
        self.featselect_featconst = featselect_featconst
        self.hv = hv
        self.selected_features = selected_features
        self.feat_sub = feat_sub
        self.glob_feat1 = glob_feat1
        self.glob_tar1 = glob_tar1
        self.new_training = new_training

    def predict_subsets(self, result=None, set_size=None, p_error=None):
        """Run the prediction on each sub-set of data on the hierarchy level.

        Parameters
        ----------
        result : list
            Contain all the coefficien and omega2 for all training data.
        set_size : list
           Size of sub-set of data/features which the model is based on.
        p_error : list
           The prediction error for plain vanilla ridge.
        """
        train_targets, train_features, index1, index2 =\
            self.hv.get_subset_data(self.index_split, self.indicies)
        if int(index1) < self.hier_level and self.featselect_featvar:
            # Reset when entering a new hierarchy level.
            p_error, set_size = [], []
            self.hier_level -= 1
        for split in range(1, 2**int(index1) + 1):
            # Take a set and train, and test on the rest.
            if split != int(index2):
                (set_size, p_error, result)\
                    = self.get_data_scale(
                    split, set_size=set_size,
                    p_error=p_error,
                    result=result)
                if (set_size and p_error and result) == []:
                    # Did not find alpha for feature-set.
                    index2 = 0
                    return (set_size, p_error, result,
                            index2, self.selected_features)
        if not self.featselect_featvar:
            # Avoid jumping into the wrong plot.
            index2 = 0
        return (set_size, p_error, result,
                index2, self.selected_features)

    def get_data_scale(self, split, set_size=None, p_error=None, result=None):
        """Get the data for each sub-set of data and scales it accordingly.

        Parameters
        ----------
        split : int
           Which sub-set od data within hierarchy level.
        result : list
            Contain all the coefficien and omega2 for all training data.
        set_size : list
           Size of sub-set of data/features which the model is based on.
        p_error : list
           The prediction error for plain vanilla ridge.
        """
        ridge = RidgeRegression()
        # Dont want the targtes to be scaled with global.
        self.s_tar, self.m_tar = None, None
        train_targets, train_features, _, _ = self.hv.get_subset_data(
            self.index_split, self.indicies)

        test_targets, test_features, _, _ =\
            self.hv.get_subset_data(self.index_split, self.indicies, split)
        # Get some statitics from data.
        (self.s_tar, self.m_tar, self.s_feat, self.m_feat,
         train_targets, train_features,
         test_features) = self.PC.scaling_data(train_features,
                                               train_targets,
                                               test_features, self.s_tar,
                                               self.m_tar, self.s_feat,
                                               self.m_feat)

        if self.featselect_featconst or self.featselect_featvar:
            # Search for the subset of features.
            if self.selected_features is None:
                if self.glob_feat1 is None and self.glob_tar1 is None:
                    #  Search with minimum data.
                    FS = feature_selection(train_features, train_targets)
                else:
                    # Search with maximum data.
                    FS = feature_selection(self.glob_feat1, self.glob_tar1)
                self.selected_features = FS.selection(self.select_limit)
                if not bool(self.selected_features):
                    return (set_size, p_error, result)

        if self.featselect_featvar:
            (set_size, p_error, result) = self.reg_feat_var(
                train_features, train_targets,
                test_features, test_targets, ridge,
                set_size, p_error,
                result)
            return (set_size, p_error, result)
        else:
            if self.featselect_featconst:
                #  Get the data for the chosen features set.
                train_features = np.take(
                    train_features,
                    self.selected_features[str(self.feat_sub)][0], axis=1)
                test_features = np.take(
                    test_features,
                    self.selected_features[str(self.feat_sub)][0], axis=1)
            set_size, p_error, result \
                = self.reg_data_var(
                    train_features, train_targets,
                    test_features, test_targets, ridge,
                    set_size, p_error, result)
            return (set_size, p_error, result)

    def reg_feat_var(self, train_features, train_targets, test_features,
                     test_targets, ridge, set_size, p_error,
                     result):
        """Regression within a dataset with varying feature.

        Parameters
        ----------
        train_features : array
            Independent data used to train the model.
        train_targets : array
            Dependent data used to train model.
        test_features : array
            Independent data used to test model.
        test_target : array
            Dependent data used to test model.
        ridge : object
           Generates the model based on the training data.
        p_error : list
           The prediction error for feature selection corresponding
           to different feature set.
        set_size : list
           Different data/feature set used for feature selection.
        result : list
            Contain all the coefficien and omega2 for all training data.
        """
        # Need reset it to get new coefficients. Could potentially fix it.
        result = []
        i = 0
        for sel_feat in self.selected_features:
            if len(result) + 1 > len(self.selected_features):
                result_x = result[i]
                i += 1
            else:
                # Get the data for the chosen feature set.
                result_x = []
            train_f = np.take(train_features,
                              self.selected_features[str(sel_feat)][0], axis=1)
            test_f = np.take(test_features,
                             self.selected_features[str(sel_feat)][0], axis=1)
            _, _, result_x \
                = self.reg_data_var(train_f, train_targets, test_f,
                                    test_targets, ridge,
                                    [np.shape(
                                     self.selected_features[str(sel_feat)][0]
                                     )[0]], p_error, result_x)
            self.new_training = True
            result.append(result_x)
            set_size.append(np.shape(
                self.selected_features[str(sel_feat)][0])[0])
        return (set_size, p_error, result)

    def reg_data_var(self, train_features, train_targets, test_features,
                     test_targets, ridge, set_size, p_error, result):
        """Ridge regression and calculation of prediction error.

        Parameters
        ----------
        train_features : array
            Independent data used to train the model.
        train_targets : array
            Dependent data used to train model.
        test_features : array
            Independent data used to test model.
        test_target : array
            Dependent data used to test model.
        ridge : object
           Generates the model based on the training data.
        set_size : list
           Size of sub-set of data/features which the model is based on.
        p_error : list
           The prediction error for plain vanilla ridge.
        result : list
            Contain all the coefficien and omega2 for all training data.
        """
        if self.new_training:
            reg_data = ridge.regularization(
                train_targets, train_features, coef=None,
                featselect_featvar=self.featselect_featvar)
        if self.new_training:
            coef = reg_data['result'][0]
        else:
            coef = result[0][4]
        data = self.PC.prediction_error(test_features, test_targets,
                                        coef, self.s_tar, self.m_tar)
        if self.new_training:
            data['result'] += reg_data['result']
            self.new_training = False
        else:
            data['result'] += (result[0][4], result[0][5])
        result.append(data['result'])
        set_size.append(data['result'][0])
        p_error.append(data['result'][1])
        return (set_size, p_error, result)

    def getstats(self):
        """Used to get features for the frequencyplots."""
        # Splitvalue of split does not matter in this case
        self.s_tar, self.m_tar = None, None

        split = 1
        train_targets, train_features, index1, index2 =\
            self.hv.get_subset_data(self.index_split, self.indicies)
        # get the data
        train_targets, train_features, _, _ = self.hv.get_subset_data(
            self.index_split, self.indicies)

        test_targets, test_features, _, _ =\
            self.hv.get_subset_data(self.index_split, self.indicies, split)
        # Get the statistics from data.
        (self.s_tar, self.m_tar, self.s_feat, self.m_feat,
         train_targets, train_features,
         test_features) = self.PC.scaling_data(train_features,
                                               train_targets,
                                               test_features, self.s_tar,
                                               self.m_tar, self.s_feat,
                                               self.m_feat)

        if self.featselect_featconst or self.featselect_featvar:
            # Search for the subset of features.
            if self.selected_features is None:
                if self.glob_feat1 is None and self.glob_tar1 is None:
                    #  Search with minimum data.
                    FS = feature_selection(train_features, train_targets)
                else:
                    # Search with maximum data.
                    FS = feature_selection(self.glob_feat1, self.glob_tar1)
                self.selected_features = FS.selection(self.select_limit)
            return self.selected_features
