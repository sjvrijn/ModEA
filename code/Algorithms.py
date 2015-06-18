__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
from .Individual import Individual
from .Parameters import Parameters
from .Selection import onePlusOneSelection

def onePlusOneES(n, fitnessFunction, budget):
    """
        Implementation of the default (1+1)-ES
        Requires the length of the vector to be optimized, the handle of a fitness function to use and the budget
    """

    parameters = Parameters(n, 1, 1)
    population = Individual(n)

    functions = {
        'recombination': lambda x: x,
        'mutation': lambda x: x,
        'selection:': lambda pop, new_pop, t: onePlusOneSelection(pop, new_pop, t, parameters),
        'mutateParameters': lambda t: parameters.oneFifthRule(t),
    }

    baseAlgorithm(population, fitnessFunction, budget, functions)



def baseAlgorithm(population, fitnessFunction, budget, functions):
    """
        Skeleton function for all ES algorithms
        Requires a population, fitness function handle, evaluation budget and the algorithm-specific functions
    """

    # Initialization
    used_budget = 0
    recombine = functions['recombination']
    mutate = functions['mutation']
    select = functions['select']
    mutateParameters = functions['mutateParameters']

    # The main evaluation loop
    while used_budget < budget:

        # Recombination
        new_population = recombine(population)
        # Mutation
        new_population = mutate(new_population)

        # Evaluation
        for individual in new_population:
            fitnessFunction(individual)

        # Selection
        population = select(new_population)
        # Parameter mutation
        mutateParameters(used_budget)

    return population