__author__ = 'Sander van Rijn <svr003@gmail.com>'

"""
This Module contains a collection of Selection operators to be used in the ES-Framework

A Selection operator accepts (mu + lambda) individuals and returns (mu) individuals
that are chosen to be the best of this generation
"""

# import numpy as np


def best(population, new_population, parameters):
    """
        Given the population, return the (mu) best
    """
    if parameters.mu < 1:
        raise Exception("best() has to return at least one individual")

    if parameters.plus_selection:
        new_population.extend(population)

    new_population.sort(key=lambda individual: individual.fitness)  # sort descending

    return new_population[:parameters.mu]


def onePlusOneSelection(population, new_population, t, parameters):
    """
        (1+1)-selection (with success history)
    """

    new_individual = new_population[0]
    individual = population[0]

    if new_individual.fitness < individual.fitness:
        parameters.best_fitness = new_individual.fitness
        result = new_population
        parameters.addToSuccessHistory(t, True)
    else:
        result = population
        parameters.addToSuccessHistory(t, False)

    return result


def onePlusOneCholeskySelection(population, new_population, _, parameters):
    """
        (1+1)-selection (with success history)
    """

    new_individual = new_population[0]
    individual = population[0]

    if new_individual.fitness < individual.fitness:
        parameters.best_fitness = new_individual.fitness
        result = new_population
        parameters.lambda_success = True
    else:
        result = population
        parameters.lambda_success = False

    return result


def onePlusOneActiveSelection(population, new_population, _, parameters):
    """
        (1+1)-selection (with success history)
    """

    new_fitness = new_population[0].fitness
    individual = population[0]

    if new_fitness < individual.fitness:
        parameters.best_fitness = new_fitness
        result = new_population
        parameters.lambda_success = True
    else:
        result = population
        parameters.lambda_success = False

    parameters.addToFitnessHistory(new_fitness)

    return result