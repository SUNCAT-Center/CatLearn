"""Modified ridge regression function from Keld Lundgaard."""
import numpy as np
from collections import defaultdict


class RidgeRegression(object):
    """Ridge regression class to find an optimal model.

    Regualization fitting can be performed with wither the loocv or
    bootstrap.632 method. The loocv method is faseter, but it is better to use
    bootstrap when there is highly correlated training data.
    """

    def __init__(self, W2=None, Vh=None, cv='loocv', Ns=100, wsteps=15,
                 rsteps=3):
        """Ridge regression setup.

        Parameters
        ----------
        W2 : list
            Singular values from the SVD.
        Vh : array
            The right unitary matrix form SVD.
        cv : string
            Define the type to CV used to find penalty term, can be 'bootstrap'
            or 'loocv'. Default is bootstrap.
        Ns : int
            Number of boostrap samples to use.
        wsteps : int
            Steps in omega2 search linespacing.
        rsteps : int
            Number of refinement steps.
        """
        self.W2 = W2
        self.Vh = Vh
        self.cv = cv
        self.Ns = Ns
        self.wsteps = wsteps
        self.rsteps = rsteps

    def predict(self, train_matrix, train_targets, test_matrix,
                test_targets=None, coefficients=None, reg=None, p=0.):
        """Function to do ridge regression predictions."""
        if coefficients is None:
            coefficients = self.get_coefficients(train_targets=train_targets,
                                                 train_features=train_matrix,
                                                 reg=reg, p=p)['coef']
        validation = []
        prediction = []
        for vec in train_matrix:
            validation.append(np.dot(coefficients, vec))
        for vec in test_matrix:
            prediction.append(np.dot(coefficients, vec))

        return validation, prediction

    def get_coefficients(self, train_targets, train_features, reg=None, p=0.):
        """Generate the omgea2 and coef value's.

        Parameters
        ----------
        train_targets : array
            Dependent data used for training.
        train_features : array
            Independent data used for training.
        reg : float
            Precomputed optimal regaluzation.
        p : float
            Define the prior function. Default is zero.
        """
        data = defaultdict(list)

        if reg is None:
            data['reg'] = self.find_optimal_regularization(train_features,
                                                           train_targets, p=p)
        data['coef'] = self.RR(train_features, train_targets, p=p,
                               omega2=data['reg'])[0]

        return data

    def find_optimal_regularization(self, X, Y, p=0.):
        """Find regualization value to minimize Expected Prediction Error.

        Parameters
        ----------
        X : array
            Feature matrix for the training data.
        Y : list
            Target data for the training sample.
        p : float
            Define the prior function. Default is zero.

        Returns
        -------
        omega2_min : float
            Regularization corresponding to the minimum EPE.
        """
        # The minimum regaluzation value
        omega2_min = float('inf')
        omega2_list = []
        epe_list = []

        # Find initial spread of omega2.
        if self.W2 is None or self.Vh is None:
            V, self.W2, self.Vh = np.linalg.svd(np.dot(X.T, X),
                                                full_matrices=True)
        # Calculate SVD for loocv.
        if self.cv is 'loocv':
            U, W, Vh = np.linalg.svd(X, full_matrices=False)

        # Set initial regularization search space.
        whigh, wlow = np.log(self.W2[0] * 2.), np.log(self.W2[-1] * 0.5)
        basesearchwidth = whigh - wlow
        omega2_range = [1e-6 * np.exp(wlow)]
        for pp in np.linspace(wlow, whigh, self.wsteps):
            omega2_range.append(np.exp(pp))
        omega2_range.append(1e6 * np.exp(whigh))

        # Find best value by successively reducing seach area for omega2.
        for s in range(self.rsteps):
            if self.cv is 'bootstrap':
                BS_res = self._bootstrap_master(X, Y, p, omega2_range, self.Ns)
                _, _, epe_list_i, _ = BS_res
            if self.cv is 'loocv':
                epe_list_i = self._LOOCV_l(X, Y, p, omega2_range, U, W)

            omega2_list += omega2_range
            epe_list += epe_list_i.tolist()

            epe_ind = np.argmin(epe_list)
            omega2_min = omega2_list[epe_ind]
            if s is 0 and epe_ind is 0 or epe_ind is len(omega2_list) - 1:
                return omega2_min

            # Update search range
            logmin_epe = np.log(omega2_min)
            basesearchwidth = 2 * basesearchwidth / (self.wsteps - 1)
            wlow = logmin_epe - basesearchwidth * 0.5
            whigh = logmin_epe + basesearchwidth * 0.5

            omega2_range = []
            for pp in np.linspace(wlow, whigh, self.wsteps):
                omega2_range.append(np.exp(pp))

        return omega2_min

    def regularization(self, train_targets, train_features, coef=None,
                       featselect_featvar=False):
        """Generate the omgea2 and coef value's.

        Parameters
        ----------
         train_targets : array
            Dependent data used for training.
        train_features : array
            Independent data used for training.
        coef : int
            List of indices in the feature database.
        """
        reg_data = {'result': None}

        if coef is None:
            b = self.find_optimal_regularization(train_features,
                                                 train_targets)
            coef = self.RR(train_features, train_targets, omega2=b,
                           featselect_featvar=featselect_featvar)[0]

            reg_data['result'] = coef, b
        return reg_data

    def RR(self, X, Y, omega2, p=0., featselect_featvar=False):
        """Ridge Regression (RR) solver.

        Cost is (Xa-y)**2 + omega2*(a-p)**2, SVD of X.T X, where T is the
        transpose V, W2, Vh = X.T*X

        Parameters
        ----------
        X : array
            Feature matrix for the training data.
        Y : list
            Target data for the training sample.
        p : float
            Define the prior function.
        omega2 : float
            Regularization strength.

        Returns
        -------
        coefs : list
            Optimal coefficients.
        neff : float
            Number of effective parameters.
        """
        # Calculating SVD
        if self.W2 is None or self.Vh is None:
            V, self.W2, self.Vh = np.linalg.svd(np.dot(X.T, X),
                                                full_matrices=True)

        R2 = np.ones(len(self.W2)) * omega2
        inv_W2_reg = (self.W2 + R2) ** -1
        XtX_reg_inv = np.dot(np.dot(self.Vh.T, np.diag(inv_W2_reg)), self.Vh)
        coefs = np.dot(XtX_reg_inv, (np.dot(X.T, Y.T) + omega2 * p))
        Neff = np.sum(self.W2 * inv_W2_reg)
        if featselect_featvar:
            self.W2 = None

        return coefs, Neff

    def _RR_preSVD(self, X, Y, p, omega2, W2, Vh):
        """Ridge Regression (RR) solver.

        Cost is (Xa-y)**2 + omega2*(a-p)**2 SVD of X.T X, where T is the
        transpose V, W2, Vh = X.T*X

        Parameters
        ----------
        X : array
            Feature matrix for the training data.
        Y : list
            Target data for the training sample.
        p : float
            Define the prior function.
        omega2 : float
            Regularization strength.
        W2 : array
            Sigular values.
        Vh : array
            Right hand side of sigular matrix for X.
        """
        R2 = np.ones(len(W2)) * omega2
        inv_W2_reg = (W2 + R2) ** -1
        XtX_reg_inv = np.dot(np.dot(Vh.T, np.diag(inv_W2_reg)), Vh)
        coefs = np.dot(XtX_reg_inv, np.dot(X.T, Y.T) + omega2 * p)

        return coefs

    def _bootstrap_master(self, X, Y, p, omega2_l, Ns):
        """Function to perform the bootstrapping.

        Parameters
        ----------
        X : array
            Feature matrix for the training data.
        Y : list
            Target data for the training sample.
        p : float
            Define the prior function.
        omega2_l : list
            List of regularization strengths to test.
        Ns : int
            Number of boostrap samples to use.
        """
        assert len(np.shape(omega2_l)) == 1
        samples = self._get_bootstrap_samples(len(Y), Ns)
        assert len(np.shape(samples)) == 2

        W2_samples, Vh_samples = self._get_samples_svd(X, samples)
        for i, omega2 in enumerate(omega2_l):
            res = self.bootstrap_calc(X, Y, p, omega2, samples,
                                      W2_samples=W2_samples,
                                      Vh_samples=Vh_samples)
            (err, ERR, EPE, coefs_samples) = res

            if i == 0:
                err_l = err
                ERR_l = ERR
                EPE_l = EPE
                coefs_samples_l = coefs_samples

            else:
                err_l = np.hstack((err_l, err))
                ERR_l = np.hstack((ERR_l, ERR))
                EPE_l = np.hstack((EPE_l, EPE))
                coefs_samples_l = np.vstack((coefs_samples_l, coefs_samples))

        return err_l, ERR_l, EPE_l, coefs_samples_l

    def _get_bootstrap_samples(self, Nd, Ns=200, seed=15):
        """Break dataset into subsets.

        Parameters
        ----------
        Nd : int
            Number of datapoints.
        Ns : int
            Number of bootstrap samples.
        """
        np.random.seed(seed)
        return np.random.random_integers(0, Nd - 1, (Ns, Nd))

    def bootstrap_calc(self, X, Y, p, omega2, samples, W2_samples, Vh_samples):
        """Calculate optimal omega2 from bootstrap.

        Parameters
        ----------
        X : array
            Feature matrix for the training data.
        Y : list
            Target data for the training sample.
        p : float
            Define the prior function.
        omega2 : float
            Regularization strength.
        samples : list
            Sample index for bootstrap.
        W2_samples : array
            Sigular values for samples.
        Vh_samples : array
            Right hand side of sigular matrix for samples.
        """
        coefs = self._RR_preSVD(X, Y, p, omega2, self.W2, self.Vh)
        err = np.sum((np.dot(X, coefs.T) - Y)**2 / len(Y))

        error_samples = []
        for i in range(len(samples)):
            W2_si = W2_samples[i]
            Vh_si = Vh_samples[:, :, i]
            X_si = X.take(samples[i], axis=0)
            Y_si = Y.take(samples[i])
            a_si = self._RR_preSVD(X_si, Y_si, p, omega2, W2_si, Vh_si)

            error = np.dot(X, a_si.T) - Y

            if i == 0:
                a_samples = a_si
                error_samples = error
            else:
                a_samples = np.vstack((a_samples, a_si))
                error_samples = np.vstack((error_samples, error))

        ERR = self._bootstrap_ERR(error_samples, samples)
        EPE = np.sqrt(0.368 * err + 0.632 * ERR)

        return err, ERR, EPE, a_samples

    def _bootstrap_ERR(self, error_samples, samples):
        """Calculate error from bootstrap.

        Parameters
        ----------
        error_samples : array
            Calculated error for samples.
        samples : list
            Sample index for bootstrap.
        """
        Nd = np.shape(error_samples)[1]
        Ns = len(samples)
        ERRi_list = np.zeros(Nd)
        for i in range(Nd):
            si = np.unique(np.where(samples == i)[0])
            nsi = np.delete(np.arange(Ns), si)
            error_nsi = error_samples.take(nsi, axis=0).take([i], axis=1)
            ERRi_list[i] = np.mean(error_nsi**2)
        ERR = np.mean(ERRi_list)

        return ERR

    def _get_samples_svd(self, X, samples):
        """Get SVD for given sample in bootstrap.

        Parameters
        ----------
        X : array
            Feature matrix for the training data.
        samples : list
            Sample index for bootstrap.
        """
        # Optimize to make loop be directly in the third dimension!
        for i, sample in enumerate(samples):
            X_si = X.take(sample.flatten(), axis=0)
            XtX_si = np.dot(X_si.T, X_si)
            V, W2, Vh = np.linalg.svd(XtX_si)
            if i == 0:
                Vh_samples = Vh
                W2_samples = W2
            else:
                Vh_samples = np.dstack((Vh_samples, Vh))
                W2_samples = np.vstack((W2_samples, W2))

        return W2_samples, Vh_samples

    def _LOOCV_l(self, X, Y, p, omega2_l, U, W):
        """Leave one out estimator for a list of regularization strengths.

        Parameters
        ----------
        X : array
            Feature matrix for the training data.
        Y : list
            Target data for the training sample.
        p : float
            Define the prior function.
        omega2_l : list
            List of regularization strengths to test.
        U : array
            Left hand side of sigular matrix for X (not form XtX).
        W : array
            Sigular values for X (not form XtX).
        """
        for i, omega2 in enumerate(omega2_l):
            LOOCV_EPE = self._LOOCV(X, Y, p, omega2, U, W)
            if i == 0:
                LOOCV_EPE_l = LOOCV_EPE
            else:
                LOOCV_EPE_l = np.hstack((LOOCV_EPE_l, LOOCV_EPE))
        return LOOCV_EPE_l

    def _LOOCV(self, X, Y, p, omega2, U, W):
        """Leave one out error estimator.

        Parameters
        ----------
        X : array
            Feature matrix for the training data.
        Y : list
            Target data for the training sample.
        p : float
            Define the prior function.
        omega2 : list
            Regularization strength.
        U : array
            Left hand side of sigular matrix for X (not form XtX).
        W : array
            Sigular values for X (not form XtX).
        """
        Y_ = Y - np.dot(X, [p] * np.shape(X)[1])

        dig1 = ((W**2 + omega2)**(-1)) * W**2
        XtX_reg_inv2 = np.dot(np.dot(U, np.diag(dig1)), U.T)
        P = np.diag(np.ones(len(Y_))) - XtX_reg_inv2
        LOOCV_EPE = len(Y_)**-1 * np.dot(np.dot(np.dot(
            np.dot(Y_.T, P), np.diag(np.diag(P)**-2)), P), Y_)

        return np.sqrt(LOOCV_EPE)
