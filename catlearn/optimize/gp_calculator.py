import numpy as np
from catlearn.regression import GaussianProcess
from scipy.spatial import distance
from catlearn.optimize.warnings import *


class GPCalculator(object):

    def __init__(self, kernel_dict=None,
                 regularization=1e-5,
                 regularization_bounds=(1e-5, 1e-5),
                 algo_opt_hyperparamters='L-BFGS-B',
                 global_optimization=False,
                 scale_data=False,
                 scale_optimizer=False,
                 opt_hyperparam=False,
                 guess_hyper=None,
                 calc_uncertainty=False):

        """GP calculator.

        Parameters
        ----------
        train_fp : list
            A list of training fingerprint vectors.
        train_target : list
            A list of training targets used to generate the predictions.
        kernel_dict : dict
            This dict can contain many other dictionarys, each one containing
            parameters for separate kernels.
            Each kernel dict contains information on a kernel such as:
            -   The 'type' key containing the name of kernel function.
            -   The hyperparameters, e.g. 'scaling', 'lengthscale', etc.
        gradients : list
            A list of gradients for all training data.
        regularization : float
            The regularization strength (smoothing function) applied to the
            covariance matrix.
        regularization_bounds : tuple
            Optional to change the bounds for the regularization.
        scale_data : boolean
            Scale the training and test features as well as target values.
            Default is False.
        """
        self.kdict = kernel_dict
        self.regularization = regularization
        self.reg_bounds = regularization_bounds
        self.scale_data = scale_data
        self.scale_optimizer = scale_optimizer
        self.global_optimization = global_optimization
        self.algo_opt_hyperparamters = algo_opt_hyperparamters
        self.opt_hyperparam = opt_hyperparam
        self.guess_hyper = guess_hyper
        self.calc_uncertainty = calc_uncertainty


    def train_process(self, train_data, target_data, gradients_data=None):
        self.trained_process = GaussianProcess(kernel_dict=self.kdict,
                               train_fp=train_data, train_target=target_data,
                               gradients=gradients_data,
                               optimize_hyperparameters=False,
                               regularization_bounds=self.reg_bounds,
                               regularization= self.regularization,
                               scale_data=self.scale_data,
                               scale_optimizer=self.scale_optimizer)
        self.train_calc = train_data.copy()
        self.target_calc = target_data.copy()
        self.gradient_calc = gradients_data.copy()
        return self.trained_process

    def opt_hyperparameters(self):
        msg = "One must train a GP before optimizing its hyper-parameters."
        assert self.trained_process, msg
        self.trained_process.optimize_hyperparameters(
        global_opt=self.global_optimization,
        algomin=self.algo_opt_hyperparamters)
        print('Hyperparameter optimization is switched on.')
        print('Optimized Hyperparameters: ', self.trained_process.theta_opt)
        print('log marginal: ', self.trained_process.log_marginal_likelihood)
        return self.trained_process

    def get_predictions(self, trained_process, test_data):
        pred = trained_process.predict(test_fp=[test_data],
        uncertainty=self.calc_uncertainty)
        prediction = np.array(pred['prediction'])
        uncertainty = None
        uncertainty_with_reg = None
        if self.calc_uncertainty is True:
            uncertainty = np.array(pred['uncertainty'])
            uncertainty_with_reg = np.array(pred['uncertainty_with_reg'])
        res = {'pred_mean':prediction,'uncertainty':uncertainty,
               'uncertainty_with_reg':uncertainty_with_reg}
        return res

    def update_hyperparameters(self, trained_process, train_data, target_data):
        # if 'width' in self.guess_hyper:
        #     new_width = distance.pdist(train_data)
        #     new_width = np.average(new_width)
        #     trained_process.kernel_dict['k1']['width'] = 0.5*new_width * \
        #     np.ones_like(trained_process.kernel_dict['k1']['width'])
        #     print('Guessed width parameter', new_width)
        if 'constant' in self.guess_hyper:
            new_constant = np.std(self.target_calc)
            trained_process.kernel_dict['k2']['const'] = new_constant
            print('Guessed constant:', new_constant)
        return trained_process
