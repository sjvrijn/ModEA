#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sander van Rijn <svr003@gmail.com>'

"""
This Module contains a collection of Recombination operators to be used in the ES-Framework

A Recombination operator accepts (mu) individuals and returns (lambda) created individuals
that are to form the new population
"""

import numpy as np
from copy import copy
from numpy import dot, mean, sqrt

def onePlusOne(population):
    """
        Utility function for 1+1 ES strategies where the recombination is merely a copy
    """
    return [copy(population[0])]


def average(pop, param):
    """
        Given the new population, return the average of the mu best individuals
    """

    avg = copy(pop[0])
    avg.dna, param.s_mean, param.sigma_mean = mean([(ind.dna, ind.mutation_vector, ind.sigma) for ind in pop],axis=0)

    new_population = [avg]
    for _ in range(param.lambda_-1):
        new_population.append(copy(avg))

    return new_population


def weighted(pop, param):
    """
        Given the population and weights, return the weighted average of the mu best
    """

    param.wcm_old = param.wcm

    offspring = np.column_stack((ind.dna for ind in pop))
    param.offspring = offspring
    param.wcm = dot(offspring, param.weights)

    new_ind = copy(pop[0])
    new_ind.dna = param.wcm
    new_population = [new_ind]
    for _ in range(param.lambda_-1):
        new_population.append(copy(new_ind))

    return new_population
