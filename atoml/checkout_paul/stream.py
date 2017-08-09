"""Placeholder for now."""
from atoml.ridge_regression import RidgeRegression
import numpy as np


class placeholder(object):
    """Used to make the hierarchey more easy to follow.

    Placeholder for now.
    """

    def __init__(self, globalscale, PC, index_split, hv, indicies,
                 hier_level, featselect_featvar, featselect_featconst,
                 s_feat, m_feat, feat_sub=15, s_tar=None, m_tar=None,
                 select_limit=None, selected_features=None, no_feature=False):
        """Function to extract raw data from the database.

        Parameters
        ----------
        globalscale : string
            Using global scaleing or not.
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
        no_feature : string
           If no feature subset found, leave the loop.
        """
        self.globalscale = globalscale
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
        self.no_feature = no_feature

    def predict_subsets(self, result=None, data_size=None, p_error=None,
                        number_feat=None, p_error_select=None, reg_store=None):
        """Run the prediction on each sub-set of data on the hierarchy level.

        Parameters
        ----------
        result : list
            Contain all the coefficien and omega2 for all training data.
        data_size : list
           Size of sub-set of data which the model is based on.
        p_error : list
           The prediction error for plain vanilla ridge.
        number_feat : list
           Different feature set used for feature selection.
        p_error_select : list
           The prediction error for feature selection corresponding
           to different feature set.
        reg_store : dict
            Saves coefficients and omega2 if to be re used.
        """
        train_targets, train_features, index1, index2 =\
            self.hv.get_subset_data(self.index_split, self.indicies)

        if int(index1) < self.hier_level:
            p_error_select, number_feat = [], []
            self.hier_level -= 1
        coef = None
        for split in range(1, 2**int(index1)+1):
            reg_data = {'result': None}
            if split != int(index2):
                if self.featselect_featvar:
                    p_error_select, number_feat = self.get_data_scale(
                                        split,
                                        p_error_select=p_error_select,
                                        number_feat=number_feat)
                else:
                    (data_size, p_error, coef, reg_data, result, reg_store) \
                     = self.get_data_scale(split, data_size=data_size,
                                           p_error=p_error, coef=coef,
                                           reg_data=reg_data, result=result,
                                           reg_store=reg_store)
                    if self.no_feature:
                        return (data_size, p_error, coef, reg_data,
                                result, reg_store, self.selected_features,
                                self.no_feature)

        if self.featselect_featvar:
            return (p_error_select, number_feat, index2,
                    self.selected_features)
        else:
            return (data_size, p_error, coef, reg_data,
                    result, reg_store, self.selected_features, self.no_feature)

    def get_data_scale(self, split, p_error_select=None, number_feat=None,
                       reg_data=None, data_size=None, p_error=None, coef=None,
                       result=None, reg_store=None):
        """Get the data for each sub-set of data and scales it accordingly.

        Parameters
        ----------
        result : list
            Contain all the coefficien and omega2 for all training data.
        data_size : list
           Size of sub-set of data which the model is based on.
        p_error : list
           The prediction error for plain vanilla ridge.
        number_feat : list
           Different feature set used for feature selection.
        p_error_select : list
           The prediction error for feature selection corresponding
           to different feature set.
        reg_store : dict
            Saves coefficients and omega2 if to be re used.
        """
        ridge = RidgeRegression()
        self.s_tar, self.s_tar = None, None
        train_targets, train_features, _, _ = self.hv.get_subset_data(
                                              self.index_split, self.indicies)

        test_targets, test_features, _, _ =\
            self.hv.get_subset_data(self.index_split, self.indicies, split)
        (self.s_tar, self.m_tar, self.s_feat, self.m_feat,
         train_targets, train_features,
         test_features) = self.PC.scaling_data(train_features,
                                               train_targets,
                                               test_features, self.s_tar,
                                               self.m_tar, self.s_feat,
                                               self.m_feat)
        if self.featselect_featvar:
            p_error_select, number_feat = self.reg_feat_var(
                                          train_features, train_targets,
                                          test_features, test_targets, ridge,
                                          p_error_select, number_feat)
            return p_error_select, number_feat
        else:
            if self.featselect_featconst:
                if self.selected_features is None:
                    from feature_selection import feature_selection
                    FS = feature_selection(train_features, train_targets)
                    self.selected_features = FS.selection(self.select_limit)
                    if not str(self.feat_sub) in self.selected_features:
                        self.no_feature = True
                        return (data_size, p_error, coef, reg_data, result,
                                reg_store)
                    else:
                        print(self.selected_features[str(self.feat_sub)][0])

                train_features = np.take(
                    train_features,
                    self.selected_features[str(self.feat_sub)][0], axis=1)
                test_features = np.take(
                   test_features,
                   self.selected_features[str(self.feat_sub)][0], axis=1)
            data_size, p_error, coef, reg_data, result, reg_store \
                = self.reg_data_var(
                                    train_features, train_targets,
                                    test_features, test_targets, ridge,
                                    data_size, p_error, coef, reg_data,
                                    result, reg_store)
            return (data_size, p_error, coef, reg_data,
                    result, reg_store)

    def reg_feat_var(self, train_features, train_targets, test_features,
                     test_targets, ridge, p_error_select, number_feat):
        """Reg and feature selection with the feature as the variable.

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
        p_error_select : list
           The prediction error for feature selection corresponding
           to different feature set.
        number_feat : list
           Different feature set used for feature selection.
        """
        if self.selected_features is None:
            from feature_selection import feature_selection
            FS = feature_selection(train_features, train_targets)
            self.selected_features = FS.selection(self.select_limit)
        for sel_feat in self.selected_features:
            train_f = np.take(train_features,
                              self.selected_features[str(sel_feat)][0], axis=1)
            test_f = np.take(test_features,
                             self.selected_features[str(sel_feat)][0], axis=1)
            reg_data = ridge.regularization(train_targets, train_f, coef=None,
                                            featselect_featvar=True)
            coef = reg_data['result'][0]
            data = self.PC.prediction_error(test_f, test_targets,
                                            coef, self.s_tar,
                                            self.m_tar)
            p_error_select.append(data['result'][1])
            number_feat.append(np.shape(
                               self.selected_features[str(sel_feat)][0])[0])
        return p_error_select, number_feat

    def reg_data_var(self, train_features, train_targets, test_features,
                     test_targets, ridge, data_size, p_error, coef,
                     reg_data, result, reg_store):
        """Reg and feature selection (default) with data as variable.

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
        data_size : list
           Size of sub-set of data which the model is based on.
        p_error : list
           The prediction error for plain vanilla ridge.
        coef : list
           The coefficient of the model.
        reg_data : dict
           Contain the coefficient and omega2 for each training sub-set.
        result : list
            Contain all the coefficien and omega2 for all training data.
        reg_store : dict
            Saves coefficients and omega2 if to be re used.
        """
        if coef is None:
            reg_data = ridge.regularization(train_targets,
                                            train_features,
                                            coef)
        if reg_data['result'] is not None:
            reg_store = reg_data['result']
            coef = reg_data['result'][0]
        data = self.PC.prediction_error(test_features, test_targets,
                                        coef, self.s_tar, self.m_tar)
        if reg_data['result'] is not None:

            data['result'] += reg_data['result']

        else:

            data['result'] += reg_store

        result.append(data['result'])
        print('data size:', data['result'][0], 'prediction error:',
              data['result'][1], 'Omega:', data['result'][5],
              'Euclidean length:', data['result'][2],
              'Pearson correlation:', data['result'][3])

        data_size.append(data['result'][0])
        p_error.append(data['result'][1])
        return (data_size, p_error, coef, reg_data, result, reg_store)
