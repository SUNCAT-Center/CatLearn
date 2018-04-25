"""Generate the learning curve."""
import numpy as np

from .data_process import data_process
from .placeholder import placeholder
from catlearn.preprocess.scaling import target_normalize


def hierarchy(cv, features, min_split, max_split, new_data=True, ridge=True,
              scale=True, globalscale=True, normalization=True,
              featselect_featvar=False, featselect_featconst=True,
              select_limit=None, feat_sub=15):
    """Start the hierarchy.

    Parameters
    ----------
    features : int
        Number of features used for regression.
    min_split : int
        Number of datasplit in the smallest sub-set.
    max_split : int
        Number of datasplit in the largest sub-set.
    new_data : string
       Use new data or the previous data.
    ridge : string
        Ridge regulazer is deafult. If False, lasso is used.
    scale : string
        If the data are supposed to be scaled or not.
    globalscale : string
        Using global scaleing or not.
    normalization : string
        If scaled, normalized or standardized. Normalized is default.
    feature_selection : string
       Using feature selection with ridge, or plain vanilla ridge.
    select_limit : int
       Up to have many number of features used for feature selection.
    """
    result, set_size, p_error = [], [], []

    # Determines how many hier_level there will be.
    hier_level = int(np.log(max_split / min_split) / np.log(2))
    PC = data_process(features, min_split, max_split, scale=scale,
                      ridge=ridge, normalization=normalization)
    selected_features = None

    if new_data:
        # Split the data into subsets.
        cv.split_index(min_split, max_split=max_split)

    # Load data back in from save file.
    index_split = cv.load_split()

    if globalscale:
        # Get all the data, and one of the largest sub-set.
        globalscaledata, glob_feat1, glob_tar1 = cv.globalscaledata(
            index_split)
        # Statistics for global scaling, and scales largest sub-set.
        s_feat, m_feat, glob_feat1 = PC.globalscaling(globalscaledata,
                                                      glob_feat1)
        data = target_normalize(glob_tar1)
        glob_tar1 = data['target']
    else:
        # Needs to be rested for each sub-set.
        s_feat, m_feat = None, None

    for indicies in reversed(index_split):
        ph = placeholder(PC, index_split, cv,
                         indicies, hier_level, featselect_featvar,
                         featselect_featconst, s_feat, m_feat,
                         select_limit=select_limit,
                         selected_features=selected_features,
                         feat_sub=feat_sub, glob_feat1=glob_feat1,
                         glob_tar1=glob_tar1)
        (set_size, p_error, result,
         index2, selected_features) = ph.predict_subsets(
            set_size=set_size,
            p_error=p_error,
            result=result)

        if int(index2) == 1:
            # When gone through all data within hier_level a plot is made
            # for varying feature with const. data size.
            featselect_featvar_plot(p_error, set_size)

        if (set_size and p_error and result) == []:
            # If no feature set is found, go to the next feature set.
            return set_size, p_error, result, PC

    return set_size, p_error, result, PC


def feature_frequency(cv, features, min_split, max_split,
                      smallest=False, new_data=True, ridge=True,
                      scale=True, globalscale=True,
                      normalization=True, featselect_featvar=False,
                      featselect_featconst=True, select_limit=None,
                      feat_sub=15):
    """Function to extract raw data from the database.

    Parameters
    ----------
    features : int
        Number of features used for regression.
    min_split : int
        Number of datasplit in the smallest sub-set.
    max_split : int
        Number of datasplit in the largest sub-set.
    new_data : string
       Use new data or the previous data.
    ridge : string
        Ridge regulazer is deafult. If False, lasso is used.
    scale : string
        If the data are supposed to be scaled or not.
    globalscale : string
        Using global scaleing or not.
    normalization : string
        If scaled, normalized or standardized. Normalized is default.
    feature_selection : string
       Using feature selection with ridge, or plain vanilla ridge.
    select_limit : int
       Up to have many number of features used for feature selection.
    """
    hier_level = int(np.log(max_split / min_split) / np.log(2))
    # Determines how many hier_level there will be.
    PC = data_process(features, min_split, max_split, scale=scale,
                      ridge=ridge, normalization=normalization)
    selected_features = None
    if new_data:
        # Split the data into subsets.
        cv.split_index(min_split, max_split=max_split)
        # Load data back in from save file.
    index_split = cv.load_split()
    indicies = (list(index_split.items())[-1])[0]

    if globalscale:
        # Get all the data, and one of the largest sub-set.
        globalscaledata, glob_feat1, glob_tar1 = cv.globalscaledata(
            index_split)
        # Statistics for global scaling, and scales largest sub-set.
        s_feat, m_feat, glob_feat1 = PC.globalscaling(globalscaledata,
                                                      glob_feat1)
        data = target_normalize(glob_tar1)
        glob_tar1 = data['target']
    if smallest:
        # Uses the smallest subset data possible for feature selection.
        glob_feat1 = None
        glob_tar1 = None
    ph = placeholder(
        PC, index_split, cv, indicies, hier_level,
        featselect_featvar, featselect_featconst, s_feat, m_feat,
        select_limit=select_limit, selected_features=selected_features,
        feat_sub=feat_sub, glob_feat1=glob_feat1, glob_tar1=glob_tar1
    )
    selected_features = ph.getstats()
    return selected_features
