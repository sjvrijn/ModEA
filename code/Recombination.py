#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

"""
This Module contains a collection of Recombination operators to be used in the ES-Framework

A Recombination operator accepts (mu) individuals and returns (lambda) created individuals
that are to form the new population
"""

import numpy as np
from copy import copy
from numpy import dot, mean, sqrt
from random import choice


def onePointCrossover(ind_a, ind_b):
    crossover_point = np.random.randint(1, len(ind_a.dna) - 2)  # -1 as randint is inclusive, -1 to not choose the last element
    ind_a[:crossover_point], ind_b[:crossover_point] = ind_b[:crossover_point], ind_a[:crossover_point]
    return ind_a, ind_b


def random(pop, param):
    """
        Create a new population by selecting random parents from the given population.
        To be used when no actual recombination occurs

        :param pop:     The population to be recombined
        :param param:   Parameter object
        :returns:       A list of lambda individuals, each a copy of a randomly chosen individual from the population
    """

    new_population = [copy(choice(pop)) for _ in range(param.lambda_)]
    return new_population


def onePlusOne(pop, _):
    """
        Utility function for 1+1 ES strategies where the recombination is merely a copy

        :param pop:     The population to be recombined
        :returns:       A copy of the first individual in the given population
    """
    return [copy(pop[0])]


def weighted(pop, param):
    """
        Given the population and weights, return the weighted average of the mu best

        :param pop:     The population to be recombined
        :param param:   Parameter object
        :returns:       A list of lambda individuals, the dna of each set to the numerical mean of the given population,
                        corrected by the weights set in the Parameters object. Set weights to 1/n for plain numerical
                        mean.
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
