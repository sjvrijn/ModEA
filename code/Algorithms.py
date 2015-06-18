__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np


def onePlusOneES(n, fitnessFunction, budget):
    """
        Implementation of the default (1+1)-ES
        Requires the length of the vector to be optimized, the handle of a fitness function to use and the budget
    """

    pass


def baseAlgorithm(population, fitnessFunction, budget, functions):
    """
        Skeleton function for all ES algorithms
        Requires a population, fitness function handle, evaluation budget and the algorithm-specific functions
    """

    used_budget = 0
    recombination = functions['recombination']
    mutation = functions['mutation']
    selection = functions['selection']

    while used_budget < budget:

        new_population = recombination(population)
        new_population = mutation(new_population)

        for individual in new_population:
            fitnessFunction(individual)

        population = selection(new_population)

    return population