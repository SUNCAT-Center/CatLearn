def modeling(self, globalscale, PC, index_split, indicies,
             hier_level, featselect_featvar, result=None,
             data_size=None, p_error=None, p_error_select=None,
             number_feat=None, s_tar=None, m_tar=None,
             coef=None, reg_store=None, select_limit=None,
             featselect_featconst=False, selected_features=None):
    """Function to extract raw data from the database.

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
    s_tar : int
        Standard deviation/(Max-Min) for training target.
    m_tar : int
        Mean for the training target.
    ridge : object
       Generates the model based on the training data.
    PC : object
       Modieifes data such as scaling.
    p_error_select : list
       The prediction error for feature selection corresponding
       to different feature set.
    number_feat : list
       Different feature set used for feature selection.
    select_limit : int
       Maximum feature set used.
    """
    if globalscale:
        globalscaledata = self.globalscaledata(index_split)
        s_feat, m_feat = PC.globalscaling(globalscaledata)
    else:
        s_feat, m_feat = None, None
    train_targets, train_features, index1, index2 =\
        self.get_subset_data(index_split, indicies)

    if int(index1) < hier_level:
        p_error_select, number_feat = [], []
        hier_level -= 1
    for split in range(1, 2**int(index1)+1):
        reg_data = {'result': None}
        if split != int(index2):
            if featselect_featvar:
                p_error_select, number_feat = self.anlysis(
                    index_split, indicies, split, s_tar, m_tar, s_feat,
                    m_feat, PC, featselect_featvar=True,
                    p_error_select=p_error_select,
                    number_feat=number_feat, select_limit=select_limit)
                return p_error_select, number_feat, index2
            else:
                if featselect_featconst:
                    (data_size, p_error, coef, reg_data, result, reg_store,
                     selected_features) = self.anlysis(index_split,
                                                       indicies, split,
                                                       s_tar, m_tar,
                                                       s_feat, m_feat, PC,
                                                       data_size=data_size,
                                                       p_error=p_error,
                                                       coef=coef,
                                                       reg_data=reg_data,
                                                       result=result,
                                                       reg_store=reg_store,
                                                       featselect_featconst
                                                       =True,
                                                       selected_features=
                                                       selected_features)
                    return (data_size, p_error, coef, reg_data, result,
                            reg_store, selected_features)


                else:
                    data_size, p_error, coef, reg_data, result, reg_store \
                        = self.anlysis(index_split, indicies, split, s_tar,
                                       m_tar, s_feat, m_feat, PC,
                                       data_size=data_size, p_error=p_error,
                                       coef=coef, reg_data=reg_data,
                                       result=result, reg_store=reg_store)
                    return (data_size, p_error, coef, reg_data, result,
                            reg_store)

def anlysis(self, index_split, indicies, split, s_tar, m_tar, s_feat,
            m_feat, PC, data_size=None, p_error=None,
            featselect_featvar=False, p_error_select=None,
            number_feat=None, select_limit=None,
            coef=None, reg_data=None, result=None, reg_store=None,
            featselect_featconst=False,
            selected_features=None):
    """Function to extract raw data from the database.

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
    s_tar : int
        Standard deviation/(Max-Min) for training target.
    m_tar : int
        Mean for the training target.
    ridge : object
       Generates the model based on the training data.
    PC : object
       Modieifes data such as scaling.
    p_error_select : list
       The prediction error for feature selection corresponding
       to different feature set.
    number_feat : list
       Different feature set used for feature selection.
    select_limit : int
       Maximum feature set used.
    """
    ridge = RidgeRegression()
    train_targets, train_features, _, _ = self.get_subset_data(index_split,
                                                               indicies)
    test_targets, test_features, _, _ =\
        self.get_subset_data(index_split, indicies,
                             split)
    (s_tar, m_tar, s_feat, m_feat,
     train_targets, train_features,
     test_features) = PC.scaling_data(train_features,
                                      train_targets,
                                      test_features, s_tar,
                                      m_tar, s_feat, m_feat)
    if featselect_featvar:
        p_error_select, number_feat =\
            self.feat_select_ridge(train_features,
                                   train_targets,
                                   test_features,
                                   test_targets, s_tar, m_tar,
                                   ridge, PC, p_error_select,
                                   number_feat, select_limit)
        return p_error_select, number_feat
    else:
        if featselect_featconst:
            if selected_features is None:
                from feature_selection import feature_selection
                FS = feature_selection(train_features, train_targets)
                selected_features = FS.selection(select_limit=select_limit)
            train_features = np.take(train_features,
                                     selected_features["15"][0], axis=1)
            test_features = np.take(test_features,
                                    selected_features["15"][0], axis=1)
            data_size, p_error, coef, reg_data, result, reg_store \
                = self.plain_vanilla_ridge(train_features, train_targets,
                                           test_features, test_targets,
                                           s_tar, m_tar, ridge, PC,
                                           data_size, p_error,
                                           coef, reg_data, result,
                                           reg_store)
            return (data_size, p_error, coef, reg_data, result, reg_store,
                    selected_features)

        else:
            data_size, p_error, coef, reg_data, result, reg_store \
               = self.plain_vanilla_ridge(train_features, train_targets,
                                          test_features, test_targets,
                                          s_tar, m_tar, ridge, PC,
                                          data_size, p_error,
                                          coef, reg_data, result,
                                          reg_store)
        return data_size, p_error, coef, reg_data, result, reg_store

def feat_select_ridge(self, train_features, train_targets, test_features,
                      test_targets, s_tar, m_tar, ridge, PC,
                      p_error_select, number_feat, select_limit):
    """Function to extract raw data from the database.

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
    s_tar : int
        Standard deviation/(Max-Min) for training target.
    m_tar : int
        Mean for the training target.
    ridge : object
       Generates the model based on the training data.
    PC : object
       Modieifes data such as scaling.
    p_error_select : list
       The prediction error for feature selection corresponding
       to different feature set.
    number_feat : list
       Different feature set used for feature selection.
    select_limit : int
       Maximum feature set used.
    """
    from feature_selection import feature_selection
    FS = feature_selection(train_features, train_targets)
    selected_features = FS.selection(select_limit=select_limit)
    for sel_feat in selected_features:
        train_f = np.take(train_features,
                          selected_features[str(sel_feat)][0], axis=1)
        test_f = np.take(test_features,
                         selected_features[str(sel_feat)][0], axis=1)
        reg_data = ridge.regularization(train_targets, train_f, coef=None,
                                        featselect_featvar=True)
        coef = reg_data['result'][0]
        data = PC.prediction_error(test_f, test_targets,
                                   coef, s_tar, m_tar)
        p_error_select.append(data['result'][1])
        number_feat.append(int(sel_feat))
    return p_error_select, number_feat

def plain_vanilla_ridge(self, train_features, train_targets, test_features,
                        test_targets, s_tar, m_tar, ridge, PC,
                        data_size, p_error, coef, reg_data, result,
                        reg_store):
    """Function to extract raw data from the database.

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
    s_tar : int
        Standard deviation/(Max-Min) for training target.
    m_tar : int
        Mean for the training target.
    ridge : object
       Generates the model based on the training data.
    PC : object
       Modieifes data such as scaling.
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

    data = PC.prediction_error(test_features, test_targets,
                               coef, s_tar, m_tar)

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
    return data_size, p_error, coef, reg_data, result, reg_store
