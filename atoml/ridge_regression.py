"""Ridge regression function from Keld Lundgaard."""
import numpy as np


def find_optimal_regularization(X, Y, p=0, Ns=100, wsteps=15, W2=None,
                                Vh=None):
    """Find optimal omega2=w value for the fitting.

    For each w2 value find the Expected Prediction Error value, which means to
    solve for the coefficients for each specific w2 value for each data point
    (leaving one data point out at the time).

    The epe is the combined difference from the solutions when leaving one data
    point out. Finally find which omega2-regualization corresponds to the
    minimum epe.

    Parameters
    ----------
    X : array
        Feature matrix for the training data.
    Y : list
        Target data for the training sample.
    p : ???
        Define the prior function.
    Ns : int
        Number of boostrap samples to use
    W2 : array
        Sigular values
    Vh : array
        Right hand side of sigular matrix for X
    """
    # The minimum regaluzation value
    omega2_min = float('inf')
    omega2_list = []
    epe_list = []

    # Find initial spread of omega2.
    if W2 is None and Vh is None:
        U, W2, Vh = np.linalg.svd(np.dot(X.T, X), full_matrices=True)
    whigh, wlow = np.log(W2[0] * 2.), np.log(W2[-1] * 0.5)
    basesearchwidth = whigh-wlow
    omega2_range = [1e-6*np.exp(wlow)]
    for pp in np.linspace(wlow, whigh, wsteps):
        omega2_range.append(np.exp(pp))
    omega2_range.append(1e6*np.exp(whigh))

    # Find best value by successively reducing seach area for omega2.
    # Range of omega2 to search for min epe
    BS_res = bootstrap_master(X, Y, p, omega2_range, Ns, W2, Vh)
    _, _, epe_list_i, _ = BS_res

    omega2_list += omega2_range
    epe_list += epe_list_i.tolist()

    omega2_min = omega2_list[np.argmin(epe_list)]
    if np.argmin(epe_list) is 0 or np.argmin(epe_list) is len(omega2_list)-1:
        return omega2_min

    # Update search range
    logmin_epe = np.log(omega2_min)
    basesearchwidth = 2*basesearchwidth/(wsteps-1)
    wlow = logmin_epe - basesearchwidth*0.5
    whigh = logmin_epe + basesearchwidth*0.5

    omega2_range = []
    for pp in np.linspace(wlow, whigh, wsteps):
        omega2_range.append(np.exp(pp))

    BS_res = bootstrap_master(X, Y, p, omega2_range, Ns, W2, Vh)
    _, _, epe_list_i, _ = BS_res

    omega2_list += omega2_range
    epe_list += epe_list_i.tolist()

    omega2_min = omega2_list[np.argmin(epe_list)]

    return omega2_min


def RR(X, Y, omega2, p=0, W2=None, Vh=None):
    """Ridge Regression (RR) solver.

    Cost is (Xa-y)**2 + omega2*(a-p)**2, SVD of X.T X, where T is the transpose
    V, W2, Vh = X.T*X

    Parameters
    ----------
    X : array
        Feature matrix for the training data.
    Y : list
        Target data for the training sample.
    p : ???
        Define the prior function.
    omega2 : float
        Regularization strength.
    W2 : array
        Sigular values
    Vh : array
        Right hand side of sigular matrix for X

    Returns
    -------
    coefs : optimal coefficients
    neff : number of effective parameters
    """
    # Calculating SVD
    if W2 is None and Vh is None:
        XtX = np.dot(X.T, X)
        try:
            V, W2, Vh = np.linalg.svd(XtX)
        except np.linalg.linalg.LinAlgError:
            raise

    R2 = np.ones(len(W2))*omega2
    inv_W2_reg = (W2 + R2)**-1
    XtX_reg_inv = np.dot(np.dot(Vh.T, np.diag(inv_W2_reg)), Vh)
    coefs = np.dot(XtX_reg_inv, (np.dot(X.T, Y.T)+omega2*p))
    Neff = np.sum(W2*inv_W2_reg)

    return coefs, Neff


def RR_preSVD(X, Y, p, omega2, W2, Vh):
    """Ridge Regression (RR) solver.

    Cost is (Xa-y)**2 + omega2*(a-p)**2 SVD of X.T X, where T is the transpose
    V, W2, Vh = X.T*X

    Parameters
    ----------
    X : datamatrix
    Y : target vector
    p : prior function
    omega2 : regularization strength
    W2 : Sigular values
    Vh : right hand side of sigular matrix for X

    Returns
    -------
    coefs : optimal coefficients
    """
    R2 = np.ones(len(W2))*omega2
    inv_W2_reg = (W2 + R2)**-1
    XtX_reg_inv = np.dot(np.dot(Vh.T, np.diag(inv_W2_reg)), Vh)
    coefs = np.dot(XtX_reg_inv, np.dot(X.T, Y.T)+omega2*p)

    return coefs


def bootstrap_master(X, Y, p, omega2_l, Ns=100, X2_W=None, X2_Vh=None):
    """Function to perform the bootstrapping."""
    assert len(np.shape(omega2_l)) == 1
    samples = get_bootstrap_samples(len(Y), Ns)
    assert len(np.shape(samples)) == 2

    # Make full SVD on all samples one time for all
    if X2_W is None and X2_Vh is None:
        X2_U, X2_W, X2_Vh = np.linalg.svd(np.dot(X.T, X), full_matrices=True)
    W2_samples, Vh_samples = get_samples_svd(X, samples)
    for i, omega2 in enumerate(omega2_l):
        res = bootstrap_calc(
            X, Y, p, omega2, samples,
            X2_W=X2_W, X2_Vh=X2_Vh,
            W2_samples=W2_samples, Vh_samples=Vh_samples
        )
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

        print('omega2', omega2, 'epe', EPE)
    return err_l, ERR_l, EPE_l, coefs_samples_l


def get_bootstrap_samples(Nd, Ns=200, seed=15):
    """Break dataset into subsets.

    Nd : number of datapoints
    Ns : number of bootstrap samples
    """
    np.random.seed(seed)
    return np.random.random_integers(0, Nd-1, (Ns, Nd))


def bootstrap_calc(X, Y, p, omega2, samples, X2_W=None, X2_Vh=None,
                   W2_samples=None, Vh_samples=None):
    """Calculate optimal omega2 from bootstrap."""
    if X2_W is None or X2_Vh is None:
        X2_U, X2_W, X2_Vh = np.linalg.svd(np.dot(X.T, X), full_matrices=True)

    if W2_samples is None or Vh_samples is None:
        W2_samples, Vh_samples = get_samples_svd(X, samples)

    coefs = RR_preSVD(X, Y, p, omega2, X2_W, X2_Vh)
    err = np.sum((np.dot(X, coefs.T)-Y)**2/len(Y))

    error_samples = []
    for i in range(len(samples)):
        W2_si = W2_samples[i]
        Vh_si = Vh_samples[:, :, i]
        X_si = X.take(samples[i], axis=0)
        Y_si = Y.take(samples[i])
        a_si = RR_preSVD(X_si, Y_si, p, omega2, W2_si, Vh_si)

        error = np.dot(X, a_si.T) - Y

        if i == 0:
            a_samples = a_si
            error_samples = error
        else:
            a_samples = np.vstack((a_samples, a_si))
            error_samples = np.vstack((error_samples, error))

    ERR = bootstrap_ERR(error_samples, samples)
    EPE = np.sqrt(0.368*err + 0.632*ERR)

    return err, ERR, EPE, a_samples


def bootstrap_ERR(error_samples, samples):
    """Calculate error from bootstrap."""
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


def get_samples_svd(X, samples):
    """Get SVD for given sample in bootstrap."""
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


def LOOCV_l(X, Y, p, omega2_l):
    """Leave one out estimator for a list of regularization strengths."""
    XtX = np.dot(X.T, X)
    V, W2, Vh = np.linalg.svd(XtX)

    for i, omega2 in enumerate(omega2_l):
        LOOCV_EPE = LOOCV(X, Y, p, omega2, W2=W2, Vh=Vh)
        if i == 0:
            LOOCV_EPE_l = LOOCV_EPE
        else:
            LOOCV_EPE_l = np.hstack((
                LOOCV_EPE_l, LOOCV_EPE))
    return LOOCV_EPE_l


def LOOCV(X, Y, p, omega2, W2=None, Vh=None):
    """Leave one out estimator.

    Implementation of http://www.anc.ed.ac.uk/rbf/intro/node43.html
    """
    Y_ = Y-np.dot(X, p)

    if W2 is None or Vh is None:
        XtX = np.dot(X.T, X)
        V, W2, Vh = np.linalg.svd(XtX)

    inv_W2_reg = (W2 + omega2)**(-1)
    XtX_reg_inv = np.dot(np.dot(Vh.T, np.diag(inv_W2_reg)), Vh)

    P = np.diag(np.ones(len(Y_))) - np.dot(X, np.dot(XtX_reg_inv, X.T))

    LOOCV_EPE = len(Y_)**-1 * np.dot(np.dot(np.dot(
        np.dot(Y_.T, P), np.diag(np.diag(P)**-2)), P), Y_)

    return np.sqrt(LOOCV_EPE)
