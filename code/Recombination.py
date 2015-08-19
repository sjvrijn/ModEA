__author__ = 'Sander van Rijn <svr003@gmail.com>'

"""
This Module contains a collection of Recombination operators to be used in the ES-Framework

A Recombination operator accepts (mu) individuals and returns (lambda) created individuals
that are to form the new population
"""

import numpy as np


def onePlusOne(population):
    """
        Utility function for 1+1 ES strategies where the recombination is merely a copy
    """
    return [population[0].getCopy()]


def average(population, parameters):
    """
        Given the new population, return the average of the mu best individuals
    """

    avg = population[0].getCopy()
    avg.dna, parameters.s_mean, parameters.sigma_mean = np.mean([(i.dna, i.last_s, i.sigma) for i in population],axis=0)

    new_population = [avg]
    for _ in range(parameters.lambda_-1):
        new_population.append(avg.getCopy())

    return new_population


def weighted(population, parameters):
    """
        Given the population and weights, return the weighted average of the mu best
    """

    mu = len(population)
    avg = population[0].getCopy()
    parameters.y_w = np.sum([population[i].last_s * parameters.weights[i] for i in range(mu)], axis=0)
    parameters.y_w_squared = np.sum([np.dot(population[i].last_s, population[i].last_s.T) * parameters.weights[i] for i in range(mu)])

    mean = np.mean([population[i].dna for i in range(mu)], axis=0)
    avg.dna = mean + parameters.y_w

    new_population = [avg]
    for _ in range(parameters.lambda_-1):
        new_population.append(avg.getCopy())

    return new_population