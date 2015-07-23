__author__ = 'Sander van Rijn <svr003@gmail.com>'

"""
This Module contains a collection of Recombination operators to be used in the ES-Framework

A Recombination operator accepts (mu) individuals and returns (lambda) created individuals
that are to form the new population
"""

import numpy as np


def average(population, parameters):
    """
        Given the new population, return the average of the mu best individuals
    """

    avg = population[0].getCopy()
    avg.dna, parameters.s_mean, parameters.sigma_mean = np.mean([(i.dna, i.last_s, i.sigma) for i in population], axis=0)

    new_population = [avg]
    for _ in range(parameters.labda-1):
        new_population.append(avg.getCopy())

    return new_population


def weighted(labda, weights, population):
    """
        Given the population and weights, return the weighted average of the mu best
    """

    mu = len(population)
    avg = population[0].getCopy()
    avg.dna = np.mean([population[i].dna * weights[i] for i in range(mu)])

    new_population = [avg]
    for _ in range(labda-1):
        new_population.append(avg.getCopy())

    return new_population