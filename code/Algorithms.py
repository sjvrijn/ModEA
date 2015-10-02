#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sander van Rijn <svr003@gmail.com>'
# External libraries
# import numpy as np
# Internal classes
from .Individual import Individual
from .Parameters import Parameters
# Internal modules
import code.Mutation as Mut
import code.Recombination as Rec
import code.Selection as Sel


def onePlusOneES(n, fitnessFunction, budget):
    """
        Implementation of the default (1+1)-ES
        Requires the length of the vector to be optimized, the handle of a fitness function to use and the budget
    """

    parameters = Parameters(n, 1, 1)
    population = [Individual(n)]
    population[0].fitness = fitnessFunction(population[0].dna)

    # We use lambda functions here to 'hide' the additional passing of parameters that are algorithm specific
    functions = {
        'recombine': lambda pop: Rec.onePlusOne(pop),  # simply copy the only existing individual and return as a list
        'mutate': lambda ind: Mut.x1(ind, parameters),
        'select': lambda pop, new_pop, t: Sel.onePlusOneSelection(pop, new_pop, t, parameters),
        'mutateParameters': lambda t: parameters.oneFifthRule(t),
    }

    return baseAlgorithm(population, fitnessFunction, budget, functions, parameters)


def CMA_ES(n, fitnessFunction, budget, mu=4, lambda_=15, elitist=False):
    """
        Implementation of a default (mu +/, lambda)-CMA-ES
        Requires the length of the vector to be optimized, the handle of a fitness function to use and the budget
    """

    parameters = Parameters(n, mu, lambda_, elitist)
    population = [Individual(n) for _ in range(mu)]

    # Artificial init: in hopes of fixing CMA-ES
    wcm = parameters.wcm
    fitness = fitnessFunction(wcm)[0]
    for individual in population:
        individual.dna = wcm
        individual.fitness = fitness

    # We use lambda functions here to 'hide' the additional passing of parameters that are algorithm specific
    functions = {
        'recombine': lambda pop: Rec.weighted(pop, parameters),
        'mutate': lambda ind: Mut.CMAMutation__(ind, parameters),
        'select': lambda pop, new_pop, _: Sel.best(pop, new_pop, parameters),
        'mutateParameters': lambda t: parameters.adaptCovarianceMatrix(t),
    }

    return baseAlgorithm(population, fitnessFunction, budget, functions, parameters)


def onePlusOneCholeskyCMAES(n, fitnessFunction, budget):
    """
        Implementation of the default (1+1)-ES
        Requires the length of the vector to be optimized, the handle of a fitness function to use and the budget
    """

    parameters = Parameters(n, 1, 1)
    population = [Individual(n)]
    population[0].fitness = fitnessFunction(population[0].dna)

    # We use lambda functions here to 'hide' the additional passing of parameters that are algorithm specific
    functions = {
        'recombine': lambda pop: Rec.onePlusOne(pop),  # simply copy the only existing individual and return as a list
        'mutate': lambda ind: Mut.choleskyCMAMutation(ind, parameters),
        'select': lambda pop, new_pop, t: Sel.onePlusOneCholeskySelection(pop, new_pop, t, parameters),
        'mutateParameters': lambda t: parameters.adaptCholeskyCovarianceMatrix(),
    }

    return baseAlgorithm(population, fitnessFunction, budget, functions, parameters)


def onePlusOneActiveCMAES(n, fitnessFunction, budget):
    """
        Implementation of the default (1+1)-ES
        Requires the length of the vector to be optimized, the handle of a fitness function to use and the budget
    """

    parameters = Parameters(n, 1, 1)
    population = [Individual(n)]
    population[0].fitness = fitnessFunction(population[0].dna)
    parameters.addToFitnessHistory(population[0].fitness)

    # We use lambda functions here to 'hide' the additional passing of parameters that are algorithm specific
    functions = {
        'recombine': lambda pop: Rec.onePlusOne(pop),  # simply copy the only existing individual and return as a list
        'mutate': lambda ind: Mut.choleskyCMAMutation(ind, parameters),
        'select': lambda pop, new_pop, t: Sel.onePlusOneActiveSelection(pop, new_pop, t, parameters),
        'mutateParameters': lambda t: parameters.adaptActiveCovarianceMatrix(),
    }

    return baseAlgorithm(population, fitnessFunction, budget, functions, parameters)


def CMSA_ES(n, fitnessFunction, budget, mu=4, lambda_=15, elitist=False):
    """
        Implementation of a default (mu +/, lambda)-CMA-ES
        Requires the length of the vector to be optimized, the handle of a fitness function to use and the budget
    """

    parameters = Parameters(n, mu, lambda_, elitist)
    population = [Individual(n) for _ in range(mu)]
    for individual in population:
        individual.fitness = fitnessFunction(individual.dna)

    # We use lambda functions here to 'hide' the additional passing of parameters that are algorithm specific
    functions = {
        'recombine': lambda pop: Rec.average(pop, parameters),
        'mutate': lambda ind: Mut.CMAMutation(ind, parameters),
        'select': lambda pop, new_pop, _: Sel.best(pop, new_pop, parameters),
        'mutateParameters': lambda t: parameters.selfAdaptCovarianceMatrix(),
    }

    return baseAlgorithm(population, fitnessFunction, budget, functions, parameters)



def baseAlgorithm(population, fitnessFunction, budget, functions, parameters):
    """
        Skeleton function for all ES algorithms
        Requires a population, fitness function handle, evaluation budget and the algorithm-specific functions

        The algorithm-specific functions should (roughly) behave as follows:

         - recombine:           The current population (mu individuals) is passed to this function,
                                and should return a new population (lambda individuals),
                                generated by some form of recombination

         - mutate:              An individual is passed to this function and should be mutated 'in-line',
                                no return is expected

         - select:              The original parents, new offspring and used budget are passed to this function,
                                and should return a new population (mu individuals) after
                                (mu+lambda) or (mu,lambda) selection

         - mutateParameters:    Mutates and/or updates all parameters where required
    """

    # TODO: allow for multiple different structures to be used; i.e. sequential VS parallel evaluation

    # Parameter tracking
    sigma_over_time = []
    best_fitness_over_time = []

    # Initialization
    used_budget = 0
    recombine = functions['recombine']
    mutate = functions['mutate']
    select = functions['select']
    mutateParameters = functions['mutateParameters']

    # Single recombination outside the eval loop to create the new population
    new_population = recombine(population)

    # The main evaluation loop
    while used_budget < budget:

        for individual in new_population:
            # Mutation
            mutate(individual)
            # Evaluation
            individual.fitness = fitnessFunction(individual.dna)
            used_budget += 1

        # Selection
        population = select(population, new_population, used_budget)
        # Recombination
        new_population = recombine(population)
        # Parameter mutation
        mutateParameters(used_budget)

        # Track parameters
        sigma_over_time.append(parameters.sigma_mean)
        best_fitness_over_time.append(population[0].fitness[0])

    return population, sigma_over_time, best_fitness_over_time