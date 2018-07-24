"""The GeneticAlgorithm class methods."""
import numpy as np
import random
import warnings
import copy
from tqdm import trange, tqdm
import multiprocessing
from catlearn.cross_validation import k_fold
from .initialize import initialize_population
from .mating import cut_and_splice
from .mutate import random_permutation, probability_remove, probability_include
from .natural_selection import population_reduction, remove_duplicates
from .convergence import Convergence
from .io import _write_data


class GeneticAlgorithm(object):
    """Genetic algorithm for parameter optimization."""

    def __init__(self, fit_func, features, targets, population_size='auto',
                 population=None, operators=None, fitness_parameters=1,
                 nsplit=2, accuracy=None, nprocs=1):
        """Initialize the genetic algorithm.

        Parameters
        ----------
        population_size : int
            Population size, same as generation size.
        fit_func : object
            User defined function to calculate fitness.
        features : array
            The feature space upon which to optimize.
        targets : array
            The targets corresponding to the feature data.
        population : list
            The current population. Default is None, will generate a random
            initial population.
        operators : list
            A list of operation functions. These are used for mating and
            mutation operations.
        fitness_parameters : int
            The number of variables to optimize. Default is a single variable.
        nsplit : int
            Number of data splits for k-fold cv.
        accuracy : int
            Number of decimal places to include when finding unique candidates
            for duplication removal. If None, duplication removel is not
            performed.
        """
        # Set parameters.
        if nprocs is None:
            self.nprocs = multiprocessing.cpu_count()
        elif isinstance(nprocs, int):
            self.nprocs = nprocs
        else:
            msg = "argument nprocs must be an integer or None."
            raise ValueError(msg)
        self.step = -1
        self.fit_func = fit_func
        self.dimension = features.shape[1]
        self.nsplit = nsplit
        self.accuracy = accuracy

        if population_size == 'auto':
            self.population_size = 2 * self.nprocs * ((7 // self.nprocs) + 1)
        elif isinstance(population_size, int):
            self.population_size = population_size
        else:
            msg = "argument population_size must be an integer or 'auto'."
            raise ValueError(msg)

        # Define the starting population.
        self.population = population
        if self.population is None:
            self.population = initialize_population(
                self.population_size, self.dimension)

        # Define the operators to use.
        self.operators = operators
        if self.operators is None:
            self.operators = [cut_and_splice, random_permutation,
                              probability_remove, probability_include]

        self.fitness_parameters = fitness_parameters
        self.pareto = False
        if self.fitness_parameters > 1:
            self.pareto = True
        if self.pareto and self.accuracy is not None:
            msg = 'Should not set an accuracy parameter for multivariable '
            msg += 'searches.'
            raise RuntimeError(msg)

        # Make some k-fold splits.
        self.features, self.targets = k_fold(
            features, targets=targets, nsplit=self.nsplit)

    def search(self, steps, natural_selection=True, convergence_operator=None,
               repeat=5, verbose=False, writefile=None):
        """Do the actual search.

        Parameters
        ----------
        steps : int
            Maximum number of steps to be taken.
        natural_selection : bool
            A flag that when set True will perform natural selection.
        convergence_operator : object
            The function to perform the convergence check. If None is passed
            then the `no_progress` function is used.
        repeat : int
            Number of repeat generations with no progress.
        verbose : bool
            If True, will print out the progress of the search. Default is
            False.
        writefile : str
            Name of a json file to save data too.

        Attributes
        ----------
        population : list
            The current population.
        fitness : list
            The fitness for the current population.
        """
        self.fitness = self._serial_iterator(self.population)
        if verbose:
            self._print_data()

        # Initialixe the convergence check.
        if convergence_operator is None:
            convergence_operator = Convergence()
            convergence_operator = convergence_operator.no_progress
        convergence_operator(self.fitness, repeat=repeat)

        for self.step in range(steps):
            offspring_list = self._new_generation()

            # Keep track of fitness for new candidates.
            new_fit = self._serial_iterator(offspring_list)
            if new_fit is None:
                break

            # Combine data sets.
            self.fitness = np.concatenate((self.fitness, new_fit))
            self.population = np.concatenate((self.population, offspring_list))

            # Perform natural selection.
            if self.accuracy is not None:
                self.population, self.fitness = remove_duplicates(
                    self.population, self.fitness, self.accuracy)
            if natural_selection:
                self.population, self.fitness = population_reduction(
                    self.population, self.fitness, self.population_size)

            if verbose:
                self._print_data()

            if writefile is not None:
                _write_data(writefile, self.population, self.fitness)

            if convergence_operator(self.fitness, repeat=repeat):
                print('CONVERGED on step {}'.format(self.step + 1))
                break

    def _new_generation(self):
        """Create a new generation of candidates.

        Returns
        -------
        offspring_list : list
            A list of paramteres for the new generation.
        """
        offspring_list = []
        for c in range(self.population_size):
            # Select an initial candidate.
            p1 = None
            while p1 is None:
                p1 = self._selection(self.population, self.fitness)

            # Select a random operator.
            operator = random.choice(self.operators)

            # First check for mating.
            if operator is cut_and_splice:
                p2 = p1
                while p2 is p1 or p2 is None:
                    p2 = self._selection(self.population, self.fitness)
                offspring_list.append(operator(p1, p2))

            # Otherwise perfrom mutation.
            else:
                offspring_list.append(operator(p1))

        return offspring_list

    def _selection(self, param_list, fit_list):
        """Perform natural selection.

        Parameters
        ----------
        param_list : list
            List of parameter sets to consider.
        fit_list : list
            list of fitnesses associated with parameter list.

        Returns
        -------
        parameter : array
            A selected set of parameters from the population.
        """
        length = len(fit_list)
        index = list(range(length))
        index_shuf = list(range(length))
        random.shuffle(index_shuf)

        # Combine data and sort.
        fit = list(zip(*sorted(zip(fit_list, index), reverse=True)))

        # Define some probability scaling.
        scale, s = [], 0
        for _ in range(length):
            s += 1. / (length + 2)
            scale.append(s)

        fit_list = list(zip(*sorted(zip(fit[1], scale), reverse=False)))[1]

        # Reorder the fitness and parameter lists.
        param_list_shuf, fit_list_shuf = [], []
        for ind in index_shuf:
            param_list_shuf.append(param_list[ind])
            fit_list_shuf.append(fit_list[ind])

        # Get random probability.
        for parameter, fitness in zip(param_list_shuf, fit_list_shuf):
            if fitness > np.random.rand(1)[0]:
                return parameter

        return None

    def _serial_iterator(self, param_list):
        """Function wrapper to calculate the fitness.

        Parameters
        ----------
        param_list : list
            List of new parameter sets to get fitness for.

        Returns
        -------
        fit : array
            The fitness based on the new parameters.
        """
        # Initialize array.
        if self.nprocs == 1:
            fit = np.zeros((len(param_list), self.fitness_parameters))

            bool_list = np.asarray(param_list, dtype=np.bool)
            for index in trange(bool_list.shape[0], leave=False,
                                desc='working generation {}'.format(self.step +
                                                                    1)):
                args = (index, bool_list, self.fitness_parameters,
                        self.features,
                        self.targets,
                        self.fit_func,
                        self.nsplit)
                calc_fit = _cross_validate(args)[1]
                fit[index] = calc_fit
        else:
            fit = self._parallel_iterator(param_list)

        if self.pareto:
            fit = self._pareto_transform(fit)

        return np.reshape(fit, (len(fit),))

    def _parallel_iterator(self, param_list_ordered):
        fit = np.zeros((len(param_list_ordered), self.fitness_parameters))

        # Order param_list according to size descending.
        nparams = [sum(p) for p in param_list_ordered]
        i = np.argsort(nparams)[::-1]
        param_list = list(np.array(param_list_ordered)[i])

        bool_list = np.asarray(param_list, dtype=np.bool)
        d = bool_list.shape[0]
        args = (
            (x, bool_list, self.fitness_parameters,
             self.features,
             self.targets,
             self.fit_func,
             self.nsplit,
             ) for x in np.arange(d))
        pool = multiprocessing.Pool(self.nprocs)
        for r in tqdm(pool.imap_unordered(
                _cross_validate, args), total=d,
                desc='nested              ', leave=False):
            fit[r[0]] = r[1]

        # Return fitness in original order.
        fit_reordered = fit[np.argsort(i), :]
        return fit_reordered

    def _pareto_transform(self, fitness):
        """Function to transform a variable with fitness to a pareto fitness.

        Parameters
        ----------
        fitness : array
            A multi-dimentional array of fitnesses.

        Returns
        -------
        result : array
            Pareto front ordering for all data points.
        """
        fit_copy = fitness.copy()
        pareto = -1.
        result = np.zeros(fit_copy.shape[0])
        while np.sum(fit_copy > -np.inf) != 0:
            bool_pareto = self._locate_pareto(fit_copy)
            result[bool_pareto] = pareto
            fit_copy[bool_pareto] = -np.inf
            pareto -= 1.

        return result

    def _locate_pareto(self, fitness):
        """Function to locate the current Pareto optimal set of solutions.

        Parameters
        ----------
        fitness : array
            A multi-dimentional array of fitnesses.

        Returns
        -------
        result : array
            Boolean array with True for data on current Pareto front.
        """
        front = np.ones(fitness.shape[0], dtype=bool)
        for index, fit in enumerate(fitness):
            if front[index]:
                front[front] = np.any(
                    fitness[front] >= fit, axis=1)

        return front

    def _print_data(self):
        """Print some output during the search."""
        msg = 'new generation, current best fitness: {0:.3f} '.format(
            np.max(self.fitness))
        msg += 'mean fitness: {0:.3f}, worst fitness: {1:.3f}'.format(
            np.mean(self.fitness), np.min(self.fitness))

        print(msg)


def _cross_validate(args):
    """Return fitness metrics from k-fold cross validation of a user defined
    function fit_func.

    Parameters
    ----------
    args : tuple
        Arguments in a tuple:

        index : int
            index.
        bool_list : list
            Booleans switching descriptors on or off.
        fitness_parameters : int
            The number of variables to optimize. Default is a single variable.
        features : array
            The feature space upon which to optimize.
        targets : array
            The targets corresponding to the feature data.
        fit_func : object
            User defined function to calculate fitness.
        nsplit : int
            Number of data splits for k-fold cv.
    """
    index = args[0]
    bool_list = args[1]
    fitness_parameters = args[2]
    features = args[3]
    targets = args[4]
    fit_func = args[5]
    nsplit = args[6]
    parameter = bool_list[index]
    calc_fit = np.zeros(fitness_parameters)
    for k in range(nsplit):
        # Sort out training and testing data.
        train_features = copy.deepcopy(features)
        train_targets = copy.deepcopy(targets)
        test_features = train_features.pop(k)[:, parameter]
        test_targets = train_targets.pop(k)

        train_features = np.concatenate(train_features,
                                        axis=0)[:, parameter]
        train_targets = np.concatenate(train_targets, axis=0)
        try:
            score = fit_func(train_features, train_targets,
                             test_features, test_targets)
            if isinstance(score, float) and fitness_parameters != 1:
                raise AssertionError("len(fit_func) != fitness_parameters")
            elif isinstance(score, list) and len(score) != fitness_parameters:
                raise AssertionError("len(fit_func) != fitness_parameters")
            calc_fit += np.array(score)
        except np.linalg.linalg.LinAlgError:
            # If there is a problem calculating fitness assign -inf.
            calc_fit += np.array(
                [float('-inf')] * fitness_parameters)
            msg = 'The fitness function is failing. Returning -inf.'
            warnings.warn(msg)
    return index, calc_fit / nsplit
