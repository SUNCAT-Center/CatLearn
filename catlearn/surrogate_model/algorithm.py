"""Class to automate building a surrogate model."""
from __future__ import print_function
import numpy as np
from tqdm import tqdm
import time
import multiprocessing


class SurrogateModel(object):
    """Surrogate modeling class. Intended for screening or optimizing in a
    predefined and finite search space."""

    def __init__(self, train_model, predict, acquisition_function,
                 train_data, target):
        """Initialize the class.

        Parameters
        ----------
        train_model : object
            function which returns a trained regression model. This function
            should either train or update the regression model.
            Parameters
            ----------

            train_fp : array
                training data matrix.
            target : list
                training target feature.

            Returns
            ----------
            model : object
                Trained model object, which can be accepted by predict.

        predict : object
            function which returns predictions, error estimates and meta data.

            Parameters
            ----------

            model : object
                model returned by train_model.
            test_fp : array
                test data matrix.
            test_target : list
                test target feature.

            Returns
            ----------
            acquition_args : list
                ordered list of arguments for aqcuisition_function.
            score : object
                arbitratry meta data for output.

        acquisition_function : object
            function which returns a list of indices of candidates to be
            acquired.

            Parameters
            ----------

            *aqcuisition_args

            Returns
            ----------
            sample : list
                List indices of candidates to be acquired.

        train_data : array
            training data matrix.
        target : list
            training target feature.
        """
        self.train_model = train_model
        self.predict = predict
        self.acquisition_function = acquisition_function
        self.train_data = train_data
        self.target = target

    def test_acquisition(self, initial_subset=None, batch_size=1, n_max=None,
                         seed=None):
        """Return an array of test results for a surrogate model.

        Parameters
        ----------
        initial_subset : list
            Row indices of data to train on in the first iteration.
        batch_size : int
            Number of training points to acquire (move from test to training)
            in every iteration.
        n_max : int
            Max number of training points to test.
        """
        if initial_subset is None:
            train_index = list(range(max(batch_size, 2)))
        else:
            train_index = initial_subset

        if n_max is None:
            n_max = len(self.target)

        if seed is not None:
            random_state = np.random.RandomState(seed)
            permute = random_state.permutation(len(self.target))
            all_train_data = np.array(self.train_data)[permute, :]
            all_target = np.array(self.target)[permute]
        else:
            all_train_data = np.array(self.train_data)
            all_target = np.array(self.target)

        output = []
        for i in tqdm(np.arange(n_max // batch_size)):
            # Setup data.
            test_index = np.delete(np.arange(len(all_train_data)),
                                   np.array(train_index))
            train_fp = np.array(all_train_data)[train_index, :]
            train_target = np.array(all_target)[train_index]
            test_fp = np.array(all_train_data)[test_index, :]
            test_target = np.array(all_target)[test_index]

            if len(test_target) == 0:
                break
            elif len(test_target) < batch_size:
                batch_size = len(test_target)
            # Train regression model.
            model = self.train_model(train_fp, train_target)

            # Make predictions.
            aqcuisition_args, score = self.predict(model, test_fp, test_target)

            # Get indices to acquire from user defined function.
            sample = self.acquisition_function(*aqcuisition_args)

            to_acquire = test_index[sample[:batch_size]]
            assert len(to_acquire) == batch_size

            # Append best candidates to be acquired.
            train_index += list(to_acquire)
            # Return meta data.
            output.append(score)
        return output

    def acquire(self, unlabeled_data, batch_size=1):
        """Return indices of datapoints to acquire, from a predefined, finite
        search space.

        Parameters
        ----------
        unlabeled_data : array
            Data matrix representing an unlabeled search space.
        initial_subset : list
            Row indices of data to train on in the first iteration.
        batch_size : int
            Number of training points to acquire (move from test to training)
            in every iteration.

        Returns
        ----------
        to_acquire : list
            Indices in order of relevance from user defined function.
        score
            User defined output from predict.
        """
        # Do regression.
        model = self.train_model(self.train_data, self.target)

        # Make predictions.
        aqcuisition_args, score = self.predict(model, unlabeled_data)

        # Get list of indices in order of relevance from user defined function.
        sample = self.acquisition_function(*aqcuisition_args)

        to_acquire = sample[:batch_size]
        assert len(to_acquire) == batch_size

        # Return best candidates and meta data.
        return list(to_acquire), score

    def ensemble_test(self, size, initial_subset=None, batch_size=1,
                      n_max=None, seed_list=None, nprocs=None):
        """Return a 3d array of test results for a surrogate model. The third
        dimension expands the ensemble of tests.

        Parameters
        ----------
        size : int
            How many tests to run.
        initial_subset : list
            Row indices of data to train on in the first iteration.
        batch_size : int
            Number of training points to acquire (move from test to training)
            in every iteration.
        n_max : int
            Max number of training points to test.
        seed_list : list
            List of integer seeds for shuffling training data.
        nprocs : int
            Number of processors for parallelization

        Returns
        ----------
        ensemble : array
            size by iterations by number of metrics array of test results.
        """
        if seed_list is None:
            seed_list = [None] * size

        ensemble = []
        if nprocs != 1:
            # First a parallel implementation.
            pool = multiprocessing.Pool(nprocs)
            args = (
                    (initial_subset, batch_size, n_max, seed_list[x],
                     self.train_data, self.target,
                     self.train_model, self.predict, self.acquisition_function)
                    for x in np.arange(size))
            for r in pool.imap_unordered(_test_acquisition, args):
                ensemble.append(r)
                # Wait to make things more stable.
                time.sleep(0.001)
            pool.close()
        else:
            for x in np.arange(size):
                arg = (initial_subset, batch_size, n_max, seed_list[x],
                       self.train_data, self.target,
                       self.train_model, self.predict,
                       self.acquisition_function)
                ensemble.append(_test_acquisition(arg))
        return ensemble


def _test_acquisition(args):
        """Return an array of test results for a surrogate model.
        Picklable implementation.

        Parameters
        ----------
        initial_subset : list
            Row indices of data to train on in the first iteration.
        batch_size : int
            Number of training points to acquire (move from test to training)
            in every iteration.
        n_max : int
            Max number of training points to test.
        seed : int
            Random seed for shuffling training data.
        train_data : array
            Training data matrix.
        target : list
            Training target feature.
        train_model : object
            function which returns a trained regression model. This function
            should either train or update the regression model.
            Parameters
            ----------

            train_fp : array
                training data matrix.
            target : list
                training target feature.

            Returns
            ----------
            model : object
                Trained model object, which can be accepted by predict.

        predict : object
            function which returns predictions, error estimates and meta data.

            Parameters
            ----------

            model : object
                model returned by train_model.
            test_fp : array
                test data matrix.
            test_target : list
                test target feature.

            Returns
            ----------
            acquition_args : list
                ordered list of arguments for aqcuisition_function.
            score : object
                arbitratry meta data for output.

        acquisition_function : object
            function which returns a list of indices of candidates to be
            acquired.

            Parameters
            ----------

            *aqcuisition_args

            Returns
            ----------
            sample : list
                Indices in order of relevance from user defined function.
        """
        initial_subset = args[0]
        batch_size = args[1]
        n_max = args[2]
        seed = args[3]
        train_data = args[4]
        target = args[5]
        train_model = args[6]
        predict = args[7]
        acquisition_function = args[8]

        if initial_subset is None:
            train_index = list(range(max(batch_size, 2)))
        else:
            train_index = initial_subset

        if n_max is None:
            n_max = len(target)

        if seed is not None:
            random_state = np.random.RandomState(seed)
            permute = random_state.permutation(len(target))
            all_train_data = np.array(train_data)[permute, :]
            all_target = np.array(target)[permute]
        else:
            all_train_data = np.array(train_data)
            all_target = np.array(target)

        output = []
        for i in np.arange(n_max // batch_size):
            # Setup data.
            test_index = np.delete(np.arange(len(all_train_data)),
                                   np.array(train_index))
            train_fp = np.array(all_train_data)[train_index, :]
            train_target = np.array(all_target)[train_index]
            test_fp = np.array(all_train_data)[test_index, :]
            test_target = np.array(all_target)[test_index]

            if len(test_target) == 0:
                break
            elif len(test_target) < batch_size:
                batch_size = len(test_target)
            # Do regression.
            model = train_model(train_fp, train_target)

            # Make predictions.
            aqcuisition_args, score = predict(model, test_fp, test_target)

            # Calculate acquisition values.
            sample = acquisition_function(*aqcuisition_args)

            to_acquire = test_index[sample[:batch_size]]
            assert len(to_acquire) == batch_size

            # Append best candidates to be acquired.
            train_index += list(to_acquire)
            # Return meta data.
            output.append(score)

        return output
