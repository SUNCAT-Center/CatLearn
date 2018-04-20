"""Feature selection with lasso."""
from __future__ import division

import numpy as np
from collections import OrderedDict

from sklearn.linear_model import LassoCV


class feature_selection(object):
    """Class made to make it possible to select features.

    Used with hierarchy cross-validation.
    """

    def __init__(self, train_features, train_targets):
        """Feature selection set-up.

        Parameters
        ----------
        train_features : array
            Independent data used to train the model.
        train_targets : array
            Dependent data used to train model.
        """
        self.train_features = train_features
        self.train_targets = train_targets

    def selection(self, select_limit):
        """Select the the feture/s that works best wtig L1."""
        feat_vec, alpha_vec, _ = self.feature_inspection()
        selected_features = OrderedDict()
        for feat in range(1, np.shape(self.train_features)[-1] + 1):
            if select_limit[0] < feat and feat < select_limit[1]:
                splits = 10
                found_alpha = False
                int_expand = 0
                while not found_alpha:
                    found_alpha, alpha = self.alpha_finder(feat_vec, alpha_vec,
                                                           feat)
                    if found_alpha:
                        # alpha = self.alpha_refinment(alpha, feat)
                        _, _, feature_index = self.feature_inspection(
                            alpha_list=[alpha])
                        selected_features[str(feat)] = feature_index
                    else:
                        if int_expand < 1:
                            feat_vec, alpha_vec, splits, int_expand =\
                                self.interval_modifier(feat_vec, alpha_vec,
                                                       feat, splits,
                                                       int_expand)
                            feat_vec, alpha_vec, _ = self.feature_inspection(
                                upper=alpha_vec[0],
                                interval=splits)
                        else:
                            found_alpha = True
        return selected_features

    def feature_inspection(self, lower=0, upper=1, interval=10**2,
                           alpha_list=None):
        """Generate interval used to search for the alpha.

        Parameters
        ----------
        lower : int
            Lower bound for the interval search.
        upper : int
            Upper bound for the interval search.
        interval: int
            Number of alphas in interval inspected.
        """
        feat_vec, alpha_vec = [], []
        if alpha_list is None:
            alpha_list = np.linspace(float(upper), float(lower), int(interval))
        for alpha in alpha_list:
            model = LassoCV(alphas=[alpha]).fit(X=self.train_features,
                                                y=self.train_targets)
            feat_vec.append(np.shape(np.nonzero(model.coef_))[1])
            alpha_vec.append(alpha)
        return feat_vec, alpha_vec, np.nonzero(model.coef_)

    def alpha_finder(self, feat_vec, alpha_vec, feat):
        """Find the alpha corresponding to the number of features.

        Parameters
        ----------
        feat_vec: list
            Features within the interval.
        alpha_vec : list
            Alphas within the interval.
        feat: int
            The group of feature searched.
        """
        found_alpha = False
        for alpha, ft in zip(alpha_vec, feat_vec):
            if ft == feat:
                found_alpha = True
                return found_alpha, alpha
        if not found_alpha:
            alpha = None
        return found_alpha, alpha

    def interval_modifier(self, feat_vec, alpha_vec, feat, splits, int_expand):
        """Modifiy the interval under inspection by reduction or expantion.

        Parameters
        ----------
        feat_vec: list
            Features within the interval.
        alpha_vec: list
            Alphas within the interval.
        feat: int
            The group of feature searched.
        splits: int
            Increase of Number of alphas
            under inspection within
            interval.
        int_expand: int
            Number of times the number of
            alphas in interval is
            increased.
        """
        index = [i for i, x in enumerate(feat_vec) if x > feat][0]
        if index > 2:
            feat_vec = feat_vec[index - 3:]
            alpha_vec = alpha_vec[index - 3:]
        if index - 1 < 3:
            splits = 2 * splits
            int_expand += 1
        return feat_vec, alpha_vec, splits, int_expand

    def alpha_refinment(self, alpha, feat, splits=10, refsteps=1, upper=1.5):
        """Find a more stringent alpha for the number of feature searched for.

        Parameters
        ----------
        alpha : int
           Initial alpha found for the nuumber of feature
           searched for. Will be used as a lower limit.
        feat : int
            The number of feature searched for.
        splits: int
            Increase of Number of alphas
            under inspection within
            interval.
        refsteps: int
           Number of refinements.
        upper:
           How many times alpha the upper
           limit should be.
        """
        for steps in range(1, refsteps + 1):
            feat_vec, alpha_vec, _ = self.feature_inspection(
                upper=upper * alpha, lower=alpha, interval=splits
            )
            found_alpha, refalpha = self.alpha_finder(
                feat_vec, alpha_vec, feat
            )
            if found_alpha:
                alpha = refalpha
        return alpha
