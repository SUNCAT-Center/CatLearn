"""The GeneticAlgorithm class methods."""
import numpy as np
import random

from .initialize import initialize_population
from .mating import cut_and_splice
from .mutate import random_permutation


class GeneticAlgorithm(object):
    """Genetic algorithm for parameter optimization."""

    def __init__(self, pop_size, fit_func, dimension, pop=None):
        """Initialize the genetic algorithm.

        Parameters
        ----------
        pop_size : int
            Population size.
        fit_func : object
            User defined function to calculate fitness.
        d_param : int
            Dimension of parameters in model.
        pop : list
            The current population. Default is None.
        """
        self.pop_size = pop_size
        self.fit_func = fit_func
        self.dimension = dimension

        self.pop = pop
        if self.pop is None:
            self.pop = initialize_population(pop_size, dimension)

    def search(self, steps):
        """Do the actual search.

        Parameters
        ----------
        steps : int
            Maximum number of steps to be taken.
        """
        self.fitness = self._get_fitness(self.pop)
        operator = [cut_and_splice, random_permutation]

        for _ in range(steps):
            offspring_list = []
            for c in range(self.pop_size):
                # Select an initial candidate.
                p1 = None
                while p1 is None:
                    p1 = self._selection(self.pop, self.fitness)

                # Select a random operator.
                op = random.choice(operator)

                # First check for mating.
                if op is cut_and_splice:
                    p2 = p1
                    while p2 is p1 or p2 is None:
                        p2 = self._selection(self.pop, self.fitness)
                    offspring_list.append(op(p1, p2))

                # Otherwise perfrom mutation.
                else:
                    offspring_list.append(op(p1))

            # Keep track of fitness for new candidates.
            new_fit = self._get_fitness(offspring_list)
            if new_fit is None:
                break

            # Combine data sets.
            extend_fit = self.fitness + new_fit
            extend_pop = np.concatenate((self.pop, offspring_list))

            # Perform natural selection.
            self._population_reduction(extend_pop, extend_fit)

    def _selection(self, param_list, fit_list):
        """Perform natural selection.

        Parameters
        ----------
        param_list : list
            List of parameter sets to consider.
        fit_list : list
            list of fitnesses associated with parameter list.
        """
        index = list(range(len(fit_list)))
        fit = list(zip(*sorted(zip(fit_list, index), reverse=True)))

        scale = []
        s = 0
        for i in fit[1]:
            s += 1 / (len(fit[1]) + 2)
            scale.append(s)

        fit_list = list(zip(*sorted(zip(fit[1], scale), reverse=False)))[1]

        param_list_shuf = []
        fit_list_shuf = []
        index_shuf = list(range(len(param_list)))
        random.shuffle(index_shuf)
        for i in index_shuf:
            param_list_shuf.append(param_list[i])
            fit_list_shuf.append(fit_list[i])

        param_list, fit_list = param_list_shuf, fit_list_shuf

        # Get random probability.
        for parameter, fitness in zip(param_list, fit_list):
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
        """
        global_details = [[i, j] for i, j in zip(pop, fit)]
        global_details.sort(key=lambda x: float(x[1]), reverse=True)

        self.pop, self.fitness, unique_list = [], [], []
        for i in global_details:
            if len(self.pop) < self.pop_size:
                if round(i[1], 5) not in unique_list:
                    self.pop.append(i[0])
                    self.fitness.append(i[1])
                    unique_list.append(round(i[1], 5))
            else:
                break

        assert len(self.pop) == len(self.fitness)

    def _get_fitness(self, param_list):
        """Function wrapper to calculate the fitness.

        Parameters
        ----------
        param_list : list
            List of new parameter sets to get fitness for.
        """
        fit = np.zeros(len(param_list))
        for index, parameter in enumerate(param_list):
            try:
                calc_fit = self.fit_func(parameter)
            except ValueError:
                calc_fit = float('-inf')

            fit[index] = calc_fit

        return fit
