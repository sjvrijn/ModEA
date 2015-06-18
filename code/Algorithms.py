__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
from .Individual import Individual
from .Mutation import x1
from .Parameters import Parameters
from .Selection import onePlusOneSelection

def onePlusOneES(n, fitnessFunction, budget):
    """
        Implementation of the default (1+1)-ES
        Requires the length of the vector to be optimized, the handle of a fitness function to use and the budget
    """

    parameters = Parameters(n, 1, 1)
    population = [Individual(n)]
    fitnessFunction(population[0])

    functions = {
        'recombine': lambda x: [x[0].getCopy()],
        'mutate': lambda x: x1(x, parameters.sigma),
        'select': lambda pop, new_pop, t: onePlusOneSelection(pop, new_pop, t, parameters),
        'mutateParameters': lambda t: parameters.oneFifthRule(t),
    }

    baseAlgorithm(population, fitnessFunction, budget, functions, parameters)



def baseAlgorithm(population, fitnessFunction, budget, functions, parameters):
    """
        Skeleton function for all ES algorithms
        Requires a population, fitness function handle, evaluation budget and the algorithm-specific functions
    """

    # Parameter tracking
    sigma_over_time = []

    # Initialization
    used_budget = 0
    recombine = functions['recombine']
    mutate = functions['mutate']
    select = functions['select']
    mutateParameters = functions['mutateParameters']

    # The main evaluation loop
    while used_budget < budget:

        # Recombination
        new_population = recombine(population)

        for individual in new_population:
            # Mutation
            mutate(individual)
            # Evaluation
            fitnessFunction(individual)
            used_budget += 1

        # Selection
        population = select(population, new_population, used_budget)
        # Parameter mutation
        mutateParameters(used_budget)

        # Track parameters
        sigma_over_time.append(parameters.sigma)

    print(sigma_over_time)
    return population