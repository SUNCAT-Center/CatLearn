"""The GeneticAlgorithm class methods."""
import numpy as np
import random
import warnings
import copy
from tqdm import trange

from atoml.cross_validation import k_fold
from .initialize import initialize_population
from .mating import cut_and_splice
from .mutate import random_permutation, probability_remove, probability_include
from .convergence import Convergence


class GeneticAlgorithm(object):
    """Genetic algorithm for parameter optimization."""

    def __init__(self, population_size, fit_func, features, targets,
                 population=None, operators=None, fitness_parameters=1,
                 nsplit=2):
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
        nslpit : int
            Number of data splits for k-fold cv.
        """
        # Set parameters.
        self.step = -1
        self.population_size = population_size
        self.fit_func = fit_func
        self.dimension = features.shape[1]
        self.nsplit = nsplit

        # Define the starting population.
        self.population = population
        if self.population is None:
            self.population = initialize_population(
                population_size, self.dimension)

        # Define the operators to use.
        self.operators = operators
        if self.operators is None:
            self.operators = [cut_and_splice, random_permutation,
                              probability_remove, probability_include]

        self.fitness_parameters = fitness_parameters
        self.pareto = False
        if self.fitness_parameters > 1:
            self.pareto = True

        # Make some k-fold splits.
        self.features, self.targets = k_fold(
            features, targets=targets, nsplit=self.nsplit)

    def search(self, steps, convergence_operator=None,
               repeat=5, verbose=False):
        """Do the actual search.

        Parameters
        ----------
        steps : int
            Maximum number of steps to be taken.
        convergence_operator : object
            The function to perform the convergence check. If None is passed
            then the `no_progress` function is used.
        repeat : int
            Number of repeat generations with no progress.
        verbose : bool
            If True, will print out the progress of the search. Default is
            False.

        Attributes
        ----------
        population : list
            The current population.
        fitness : list
            The fitness for the current population.
        """
        self.fitness = self._get_fitness(self.population)
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
            new_fit = self._get_fitness(offspring_list)
            if new_fit is None:
                break

            # Combine data sets.
            extend_fit = np.concatenate((self.fitness, new_fit))
            extend_pop = np.concatenate((self.population, offspring_list))

            # Perform natural selection.
            self._population_reduction(extend_pop, extend_fit)

            if verbose:
                self._print_data()

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
            s += 1 / (length + 2)
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

    def _population_reduction(self, pop, fit):
        """Method to reduce population size to constant.

        Parameters
        ----------
        pop : list
            Extended population.
        fit : list
            Extended fitness assignment.

        Attributes
        ----------
        pop : list
            The population after natural selection.
        fitness : list
            The fitness for the current population.
        """
        # Combine parameters and sort.
        global_details = [[i, j] for i, j in zip(pop, fit)]
        global_details.sort(key=lambda x: float(x[1]), reverse=True)

        # Reinitialize everything as empty list.
        self.population, self.fitness, unique_list = [], [], []

        # Fill the lists with current best candidates.
        for i in global_details:
            if len(self.population) < self.population_size:
                # Round to some tolerance to make sure unique fitness.
                if round(i[1], 5) not in unique_list and not self.pareto:
                    self.population.append(i[0])
                    self.fitness.append(i[1])
                    unique_list.append(round(i[1], 5))
                else:
                    self.population.append(i[0])
                    self.fitness.append(i[1])
            else:
                break

    def _get_fitness(self, param_list):
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
        fit = np.zeros((len(param_list), self.fitness_parameters))

        bool_list = np.asarray(param_list, dtype=np.bool)
        for index in trange(bool_list.shape[0], leave=False,
                            desc='working generration {}'.format(self.step + 1)
                            ):
            parameter = bool_list[index]
            calc_fit = np.zeros(self.fitness_parameters)
            for k in range(self.nsplit):
                # Sort out training and testing data.
                train_features = copy.deepcopy(self.features)
                train_targets = copy.deepcopy(self.targets)
                test_features = train_features.pop(k)[:, parameter]
                test_targets = train_targets.pop(k)

                train_features = np.concatenate(train_features,
                                                axis=0)[:, parameter]
                train_targets = np.concatenate(train_targets, axis=0)
                try:
                    calc_fit += np.array(self.fit_func(
                        train_features, train_targets, test_features,
                        test_targets))
                except ValueError:
                    # If there is a problem calculating fitness assign -inf.
                    calc_fit += np.array(
                        [float('-inf')] * self.fitness_parameters)
                    msg = 'The fitness function is failing. Returning -inf.'
                    warnings.warn(msg)

            fit[index] = calc_fit / float(self.nsplit)

        if self.pareto:
            fit = self._pareto_trainsform(fit)

        return np.reshape(fit, (len(fit),))

    def _pareto_trainsform(self, fitness):
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
