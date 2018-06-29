import numpy as np
from catlearn.regression import GaussianProcess
from catlearn.optimize.constraints import apply_mask_ase_constraints


class GPCalculator(object):

    def __init__(self, kernel_dict=None,
                 regularization=1e-3,
                 regularization_bounds=((1e-5, 1e-3),),
                 algo_opt_hyperparamters='L-BFGS-B',
                 scale_data=False,
                 scale_optimizer=False,
                 opt_hyperparam=False,
                 guess_hyper=None,
                 calc_uncertainty=False):

        """GP calculator.

        Parameters
        ----------
        kernel_dict : dict
            This dict can contain many other dictionarys, each one containing
            parameters for separate kernels.
            Each kernel dict contains information on a kernel such as:
            -   The 'type' key containing the name of kernel function.
            -   The hyperparameters, e.g. 'scaling', 'lengthscale', etc.
        regularization : float
            The regularization strength (smoothing function) applied to the
            covariance matrix.
        regularization_bounds : tuple
            Optional to change the bounds for the regularization.
        scale_data : bool
            Scale the training and test features as well as target values.
            Default is False.
        opt_hyperparam : bool
            If True it will optimize the hyperparameters.
        calc_uncertainty : bool
            If True it will calculate the predicted uncertainty.

        """
        self.kdict = kernel_dict
        self.regularization = regularization
        self.reg_bounds = regularization_bounds
        self.scale_data = scale_data
        self.scale_optimizer = scale_optimizer
        self.global_optimization = False
        self.algo_opt_hyperparamters = algo_opt_hyperparamters
        self.opt_hyperparam = opt_hyperparam
        self.guess_hyper = guess_hyper
        self.calc_uncertainty = calc_uncertainty

        self.train_calc = None
        self.target_calc = None
        self.gradient_calc = None

        self.trained_process = None

    def train_process(self, train_data, target_data, gradients_data=None):

        """Train Gaussian Process.

        Parameters
        ----------

        train_data : list
            A list of training fingerprint vectors.
        target_data : list
            A list of training targets used to generate the predictions.
        gradients_data : list
            A list of first derivative observations.

        Returns
        -------
        Trained process.
        """

        self.trained_process = GaussianProcess(
                                   kernel_dict=self.kdict,
                                   train_fp=train_data,
                                   train_target=target_data,
                                   gradients=gradients_data,
                                   optimize_hyperparameters=False,
                                   regularization_bounds=self.reg_bounds,
                                   regularization=self.regularization,
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
        print('Log marginal likelihood: ',
              self.trained_process.log_marginal_likelihood)
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
        res = {'pred_mean': prediction, 'uncertainty': uncertainty,
               'uncertainty_with_reg': uncertainty_with_reg}
        return res

    def update_hyperparameters(self, trained_process):
        if 'constant' in self.guess_hyper:
            new_constant = np.std(self.target_calc)
            trained_process.kernel_dict['k2']['const'] = new_constant
            print('Guessed constant:', new_constant)
        return trained_process


def train_ml_process(list_train, list_targets, list_gradients,
                     index_constraints, ml_calculator, scaling_targets):
    """Trains a machine learning process.

    Parameters (self):

    Parameters
    ----------
    list_train : ndarray
        List of positions (in Cartesian).
    list_targets : ndarray
        List of energies.
    list_gradients : ndarray
        List of gradients.
    index_constraints : ndarray
        List of constraints constraints generated
        previously. In order to 'hide' fixed atoms to the ML
        algorithm we create a constraints mask. This
        allows to reduce the size of the training
        features (avoids creating a large covariance matrix).
    ml_calculator: object
        Machine learning calculator. See above.
    scaling_targets: float
        Scaling of the train targets.
    Returns
    --------
    dictionary containing:
        scaling_targets : scaling for the energies of the training set.
        trained_process : trained process ready for testing.
        ml_calc : returns the ML calculator (if changes have been made,
              e.g. hyperparamter optimization).


    """

    if index_constraints is not None:
        list_train = apply_mask_ase_constraints(
                                   list_to_mask=list_train,
                                   mask_index=index_constraints)[1]
        list_gradients = apply_mask_ase_constraints(
                                        list_to_mask=list_gradients,
                                        mask_index=index_constraints)[1]

    # Scale energies.

    list_targets = list_targets - np.ones_like(list_targets) * scaling_targets

    trained_process = ml_calculator.train_process(
                                                 train_data=list_train,
                                                 target_data=list_targets,
                                                 gradients_data=list_gradients)

    if ml_calculator.__dict__['opt_hyperparam'] is True:
        ml_calculator.opt_hyperparameters()

    return {'ml_calc': ml_calculator, 'trained_process': trained_process}
