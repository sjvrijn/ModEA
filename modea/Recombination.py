#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This Module contains a collection of Recombination operators

A Recombination operator accepts (mu) individuals and returns (lambda) created individuals
that are to form the new population
"""
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'


import numpy as np
from copy import copy
from numpy import dot
from random import choice


def onePointCrossover(ind_a, ind_b):
    """
        Perform one-point crossover between two individuals.

        :param ind_a:   An individual
        :param ind_b:   Another individual
        :returns:       The original individuals, whose genotype has been modified inline
    """
    # -1 as randint is inclusive, -1 to not choose the last element
    crossover_point = np.random.randint(1, len(ind_a.genotype) - 2)
    ind_a.genotype[:crossover_point], ind_b.genotype[:crossover_point] = ind_b.genotype[:crossover_point], ind_a.genotype[:crossover_point]
    return ind_a, ind_b


def random(pop, param):
    """
        Create a new population by selecting random parents from the given population.
        To be used when no actual recombination occurs

        :param pop:     The population to be recombined
        :param param:   :class:`~modea.Parameters.Parameters` object
        :returns:       A list of lambda individuals, each a copy of a randomly chosen individual from the population
    """

    new_population = [copy(choice(pop)) for _ in range(param.lambda_)]
    return new_population


def onePlusOne(pop, param):
    """
        Utility function for 1+1 ES strategies where the recombination is merely a copy

        :param pop:     The population to be recombined
        :param param:   :class:`~modea.Parameters.Parameters` object
        :returns:       A copy of the first individual in the given population
    """
    return [copy(pop[0])]


def weighted(pop, param):
    """
        Returns a new set of individuals whose genotype is initialized to the weighted average of that
        of the given population, using the weights stored in the :class:`~modea.Parameters.Parameters` object.
        Set weights to 1/n to simply use the arithmetic mean

        :param pop:     The population to be recombined
        :param param:   :class:`~modea.Parameters.Parameters` object, of which ``param.weights``
                        will be used to calculate the weighted average
        :returns:       A list of lambda individuals, with as genotype the weighted average of the given population.
    """

    param.wcm_old = param.wcm

    offspring = np.column_stack([ind.genotype for ind in pop])
    param.offspring = offspring
    param.wcm = dot(offspring, param.weights)

    new_ind = copy(pop[0])
    new_ind.genotype = param.wcm
    new_population = [new_ind]
    for _ in range(int(param.lambda_-1)):
        new_population.append(copy(new_ind))

    return new_population


def MIES_recombine(pop, param):
    """
        Returns a new set of individuals whose genotype is determined according to
        the Mixed-Integer ES by Rui Li.

        :param pop:     The population to be recombined
        :param param:   :class:`~modea.Parameters.Parameters` object
        :returns:       A list of lambda individuals, with as genotype the weighted average of the given population.
    """
    new_ind = copy(pop[0])
    new_population = [new_ind]
    reco = 1  # TODO: Remove or store in Parameters

    for _ in range(param.lambda_-1):
        # Select random individual from the current parent population
        c1 = np.random.random_integers(0, param.mu_int-1)
        c2 = np.random.random_integers(0, param.mu_int-1)

        if reco == 1:
            new_population.append(copy(pop[c1]))  # Optional: replace by copy(choice(pop))
        elif reco == 0:
            new_ind = copy(pop[c1])
            new_ind.fitness = None  # This is a choice
            new_ind.genotype += pop[c2].genotype
            new_ind.genotype //= 2
            new_population.append(new_ind)
        elif reco > 1:
            x = (choice(range(1, 10000))/10000)
            if x > 0.5:
                new_population.append(copy(pop[c1]))
            else:
                new_population.append(copy(pop[c2]))

    return new_population
