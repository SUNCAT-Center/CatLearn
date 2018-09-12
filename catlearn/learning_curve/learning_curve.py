"""Generate the learning curve."""
import numpy as np

import time
import multiprocessing
from tqdm import trange

from .data_process import data_process
from .placeholder import placeholder
from catlearn.preprocess.scaling import target_normalize


class LearningCurve(object):
    """Learning curve class. Test a model while varying
    the density of the training data."""
    def __init__(self, nprocs=1):
        """Initialize the class.

        Parameters
        ----------
        nprocs : int
            Number of processers used in parallel implementation. Default is 1
            e.g. serial.
        """
        self.nprocs = nprocs

    def run(self, model, train, target, test, test_target,
            step=1, min_data=2):
        """Evaluate a model versus training data size.

        Parameters
        ----------
        model : object
            A function that will train or load a regression model or classifier
            and make predictions for testing. model should accept
            the parameters:

                train_features : array
                test_features : array
                train_targets : list
                test_targets : list

            model should return either a float or a list of floats. The float
            or the first value of the list will be used as the fitness score.
        train : array
            An n, d array of training examples.
        targets : list
            A list of the target values.
        test : array
            An n, d array of test data.
        test targets : list
            A list of the test target values.
        step : int
            Incrementent the data set size by this many examples.
        min_data : int
            Smallest number of training examples to test.

        Returns
        -------
        output : array
            Each row is the output from the model object.
        """
        n, d = np.shape(train)
        # Get total number of iterations
        total = (n - min_data) // step
        output = []
        # Iterate through the data subset.
        if self.nprocs != 1:
            # First a parallel implementation.
            pool = multiprocessing.Pool(self.nprocs)
            tasks = np.arange(total)
            args = (
                (x, step, train, test, target,
                 test_target, model) for x in tasks)
            for r in pool.imap_unordered(_single_test, args):
                output.append(r)
                # Wait to make things more stable.
                time.sleep(0.001)
            pool.close()
        else:
            # Then a more verbose serial implementation.
            for x in trange(
                    total,
                    desc='nested              ', leave=False):
                args = (x, step, train, test,
                        target, test_target, model)
                r = _single_test(args)
                output.append(r)
        return output


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
            # When gone through all data within hier_level a file is saved
            # for varying feature with const. data size.
            arr = np.hstack([np.vstack(), np.vstack])
            np.save('hierarchy_' + str() + '.npy', arr)

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


def _single_test(args):
    """Run a model on a subset of training data with a fixed test set.

    Return the output of a function specified by the last argument.

    Parameters
    ----------
    args : tuple
        Parameters and data to be passed to model.

        args[0] : int
            Increment.
        args[1] : int
            Step size. args[1] * args[0] training examples will be passed
            to the regression model.
        args[2] : array
            An n, d array of training examples.
        args[3] : list
            A list of the target values.
        args[4] : array
            An n, d array of test data.
        args[5] : list
            A list of the test target values.
        args[6] : object
            custom function for testing a predictive model.
            Must accept 4 parameters, which are args[2:5].
    """
    # Unpack args tuple.
    x = args[0]
    n = x * args[1]
    train_features = args[2]
    test = args[3]
    train_targets = args[4]
    test_targets = args[5]
    model = args[6]

    # Delete required subset of training examples.
    train = train_features[-n:, :]
    targets = train_targets[-n:]

    # Calculate the error or other metrics from the model.
    result = model(train, targets, test, test_targets)
    return result
