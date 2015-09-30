#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sander van Rijn <svr003@gmail.com>'

"""
This Module contains a collection of Recombination operators to be used in the ES-Framework

A Recombination operator accepts (mu) individuals and returns (lambda) created individuals
that are to form the new population
"""

import numpy as np
from numpy import dot, mean, sqrt, sum

def onePlusOne(population):
    """
        Utility function for 1+1 ES strategies where the recombination is merely a copy
    """
    return [population[0].getCopy()]


def average(pop, param):
    """
        Given the new population, return the average of the mu best individuals
    """

    avg = pop[0].getCopy()
    avg.dna, param.s_mean, param.sigma_mean = mean([(ind.dna, ind.mutation_vector, ind.sigma) for ind in pop],axis=0)

    new_population = [avg]
    for _ in range(param.lambda_-1):
        new_population.append(avg.getCopy())

    return new_population


def weighted(pop, param):
    """
        Given the population and weights, return the weighted average of the mu best
    """

    weights = param.weights

    mu = len(pop)
    avg = pop[0].getCopy()
    param.weighted_mutation_vector = sum([pop[i].mutation_vector * weights[i] for i in range(mu)], axis=0)
    print("vec", param.B, "val", param.D.T, "z", [pop[i].mutation_vector.T for i in range(mu)], sep='\n')
    print("mut vector S", param.weighted_mutation_vector.T)
    param.y_w_squared = sum([dot(pop[i].mutation_vector, pop[i].mutation_vector.T) * weights[i] for i in range(mu)], axis=0)

    # mean = np.mean([population[i].dna for i in range(mu)], axis=0)
    # avg.dna = mean + parameters.weighted_mutation_vector

    avg.dna = mean([pop[i].dna * weights[i] for i in range(mu)])

    new_population = [avg]
    for _ in range(param.lambda_-1):
        new_population.append(avg.getCopy())

    return new_population