import unittest
import numpy as np

from catlearn.regression import GaussianProcess
from catlearn.regression.gpfunctions.kernels import gaussian_kernel

n_train = 5
D = 10
n_test = 20

train = np.random.rand(n_train, D) + np.indices((n_train, D))[0]
target = np.random.rand(n_train, 1) + np.indices((n_train, 1))[0]
gradients = np.random.rand(n_train, D) + np.indices((n_train, D))[0]
test = np.random.rand(n_test, D) + np.indices((n_test, D))[0]
width = np.random.rand(D)
scaling = np.random.rand()
constant = np.random.rand()


class TestGaussianKernel(unittest.TestCase):
    def test_cinv_is_good(self):
        k_dict = {'k1':
                  {'type': 'gaussian', 'width': width, 'scaling': scaling}
                  }
        gp = GaussianProcess(
            kernel_dict=k_dict, train_fp=train, train_target=target,
            gradients=None, scale_data=True, optimize_hyperparameters=False)
        gp_grad = GaussianProcess(
            kernel_dict=k_dict, train_fp=train, train_target=target,
            gradients=gradients, scale_data=True,
            optimize_hyperparameters=False)

        print('Checking size of covariance matrix (K).')
        self.assertEqual(np.shape(gp.cinv), (len(train), len(train)))

        print('Checking size of covarience matrix with gradients (Ktilde).')
        self.assertEqual(np.shape(gp_grad.cinv),
                         (len(train) + len(train) * np.shape(train)[1],
                          len(train) + len(train) * np.shape(train)[1]))

        print('Comparing bigK and bigKtilde.')
        cov = np.linalg.inv(gp.cinv)
        cov_grad = np.linalg.inv(gp_grad.cinv)

        np.testing.assert_array_almost_equal(
            cov, cov_grad[0:len(train), 0:len(train)], decimal=10)

    def test_kernel_train(self):
        bigK = gaussian_kernel(theta=width, log_scale=False, m1=train,
                               eval_gradients=False)
        bigKtilde = gaussian_kernel(theta=width, log_scale=False, m1=train,
                                    eval_gradients=True)
        bigKdg = bigKtilde[n_train:n_train + n_train * D, :n_train]
        bigKgd = bigKtilde[:n_train, n_train:n_train + n_train * D]
        bigKdd = bigKtilde[n_train:n_train + n_train * D,
                           n_train:n_train + n_train * D]

        print('Comparing bigK (without gradients) and bigK in bigKtilde.')
        np.testing.assert_array_equal(bigK, bigKtilde[:n_train, :n_train])

        print('Checking block matrices bigKgd and bigKdg.')
        np.testing.assert_array_equal(bigKgd, bigKdg.T)

        # Get two random positions.
        i_pos = np.random.randint(len(train))
        j_pos = np.random.randint(len(train))
        d = train[i_pos] - train[j_pos]

        print('Checking bigK in bigKtilde.')
        bigK_math = (np.exp(-np.linalg.norm(d / width)**2 / 2))
        np.testing.assert_array_almost_equal(bigK_math, bigK[j_pos, i_pos],
                                             decimal=15)

        print('Checking bigKgd in bigKtilde.')
        bigKgd_math = (width**(-2) * d * np.exp(
            -np.linalg.norm(d / width)**2 / 2))
        np.testing.assert_array_almost_equal(
            bigKgd_math, bigKgd[i_pos:i_pos + 1, j_pos * D:(j_pos + 1) * D][0],
            decimal=15)

        print('Checking bigKdd in bigKtilde.')
        bigKdd_math = (
            np.identity(len(d)) * width**(-2) - np.outer(
                width**(-2) * d, (width**(-2) * d).T)) * np.exp(
                    -np.linalg.norm(d / width)**2 / 2)
        np.testing.assert_array_almost_equal(bigKdd_math, bigKdd[j_pos * D:(
            j_pos + 1) * D, i_pos * D:(i_pos + 1) * D], decimal=15)

    def test_kernel_prediction(self):
        k = gaussian_kernel(theta=width, log_scale=False, m1=train, m2=test,
                            eval_gradients=False)
        ktilde = gaussian_kernel(theta=width, log_scale=False, m1=train,
                                 m2=test, eval_gradients=True)
        print('Comparing k (without gradients) and k in ktilde.')
        np.testing.assert_array_equal(k, ktilde[:, :n_test])

        # Get two random positions.
        i_pos = np.random.randint(len(train))
        j_pos = np.random.randint(len(test))
        d = train[i_pos] - test[j_pos]

        print('Checking k in ktilde.')
        k_math = np.exp(-np.linalg.norm(d / width)**2 / 2)
        np.testing.assert_array_almost_equal(k_math, ktilde[i_pos, j_pos],
                                             decimal=15)

        print('Checking kgd in ktilde.')
        kgd_math = width**(-2) * d * np.exp(-np.linalg.norm(d / width)**2 / 2)
        np.testing.assert_array_almost_equal(
            kgd_math, ktilde[
                i_pos, n_test + D * j_pos:(n_test + D * j_pos) + D],
            decimal=15)

    def test_hyperparameter_opt(self):
        k_dict = {'k1': {'type': 'gaussian',
                         'width': width, 'scaling': scaling}}
        gp_hyp = GaussianProcess(
            kernel_dict=k_dict, train_fp=train, train_target=target,
            gradients=gradients, scale_data=True,
            optimize_hyperparameters=False, regularization=1e-3)
        gp_hyp.optimize_hyperparameters(algomin='L-BFGS-B', global_opt=False)
        bigKtilde_hyp = np.linalg.inv(gp_hyp.cinv)
        bigKdg_hyp = bigKtilde_hyp[n_train:n_train + n_train * D, :n_train]
        bigKgd_hyp = bigKtilde_hyp[:n_train, n_train:n_train + n_train * D]

        print('Checking block matrices bigKgd and bigKdg (opt. hyper.).')
        np.testing.assert_array_almost_equal(bigKdg_hyp, bigKgd_hyp.T,
                                             decimal=10)


if __name__ == '__main__':
    unittest.main()
