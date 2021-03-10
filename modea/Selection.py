#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains a collection of Selection operators.

Selection accepts (mu + lambda) individuals and returns (mu) individuals
that are chosen to be the best of this generation according to which selection
module is chosen.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
from scipy import stats
from modea import Utils


def bestGA(population, new_population, param):
    """
        Given the population, return the (mu) best

        :param population:      List of :class:`~modea.Individual.MixedIntIndividual` objects containing the previous generation
        :param new_population:  List of :class:`~modea.Individual.MixedIntIndividual` objects containing the new generation
        :param param:           :class:`~modea.Parameters.Parameters` object for storing all parameters, options, etc.
        :returns:               A slice of the sorted new_population list.
    """
    if param.elitist:
        new_population.extend(population)
    new_population.sort(key=Utils.getFitness)  # sort ascending

    return new_population[:param.mu_int]


def best(population, new_population, param):
    """
        Given the population, return the (mu) best. Also performs some 'housekeeping' for the CMA-ES by collecting
        all genotypes and most recent mutation vectors and storing them in the ``param`` object.

        :param population:      List of :class:`~modea.Individual.FloatIndividual` objects containing the previous generation
        :param new_population:  List of :class:`~modea.Individual.FloatIndividual` objects containing the new generation
        :param param:           :class:`~modea.Parameters.Parameters` object for storing all parameters, options, etc.
        :returns:               A slice of the sorted new_population list.
    """
    if param.elitist:
        new_population.extend(population)

    new_population.sort(key=Utils.getFitness)  # sort ascending

    # TODO: REMOVE THESE OPERATIONS FROM THIS FUNCTION, UNEXPECTED/UNDOCUMENTED FUNCTIONALITY
    offspring = np.column_stack([ind.genotype for ind in new_population])  # Update to use the actual mutations
    offset = np.column_stack([ind.mutation_vector for ind in new_population])
    param.all_offspring = offspring
    param.offset = offset

    return new_population[:param.mu_int]


def pairwise(population, new_population, param):
    """
        Perform a selection on individuals in a population per pair, before letting :func:`~best`
        make the final selection. Intended for use with a :class:`~modea.Sampling.MirroredSampling`
        sampler to prevent step-size bias.

        Assumes that new_population contains pairs as [P1_a, P1_b, P2_a, P2_b, etc ... ]

        :param population:      List of :class:`~modea.Individual.FloatIndividual` objects containing the previous generation
        :param new_population:  List of :class:`~modea.Individual.FloatIndividual` objects containing the new generation
        :param param:           :class:`~modea.Parameters.Parameters` object for storing all parameters, options, etc.
        :returns:               A slice of the sorted new_population list.
    """
    pairwise_filtered = []
    num_pairs = len(new_population) // 2

    if len(new_population) % 2 != 0:
        # raise Exception("Error: attempting to perform pairwise selection on an odd number of individuals")
        pairwise_filtered.append(new_population[-1])  # TODO FIXME: TEMP FIX, OFTEN INCORRECT

    # Select the best (=lowest) fitness for each consecutive pair of individuals
    for i in range(num_pairs):
        index = i*2
        if new_population[index].fitness < new_population[index+1].fitness:
            pairwise_filtered.append(new_population[index])
        else:
            pairwise_filtered.append(new_population[index+1])

    # After pairwise filtering, we can re-use the regular selection function
    return best(population, pairwise_filtered, param)


def roulette(population, new_population, param, force_unique=False):
    """
        Given the population, return mu individuals, selected by roulette, using 1/fitness as probability

        :param population:      List of :class:`~modea.Individual.FloatIndividual` objects containing the previous generation
        :param new_population:  List of :class:`~modea.Individual.FloatIndividual` objects containing the new generation
        :param param:           :class:`~modea.Parameters.Parameters` object for storing all parameters, options, etc.
        :param force_unique:    Determine if an individual from the original population may be selected multiple times
        :returns:               A slice of the sorted new_population list.
    """
    if param.elitist:
        new_population.extend(population)

    new_population.sort(key=Utils.getFitness)  # sort descending
    offspring = np.column_stack([ind.genotype for ind in new_population])
    param.all_offspring = offspring

    # TODO: warning with negative fitness values?
    # Use normalized 1/fitness as probability for picking a certain individual
    norm_inverses = np.array([1/abs(ind.fitness) for ind in new_population])  # We take the absolute just to be sure it works
    norm_inverses /= sum(norm_inverses)

    # Create a discrete sampler using the normalized 1/fitness values as probabilities
    roulette_sampler = stats.rv_discrete(name='roulette', values=(list(range(len(new_population))), norm_inverses))

    if force_unique:
        indices = set()
        while len(indices) < param.mu_int:
            to_be_sampled = param.mu_int - len(indices)
            # Draw <to_be_sampled> samples from the defined distribution
            sample = roulette_sampler.rvs(size=to_be_sampled)
            indices.update(sample)
    else:
        indices = roulette_sampler.rvs(size=param.mu_int)

    return [new_population[index] for index in indices]


def onePlusOneSelection(population, new_population, t, param):
    """
        (1+1)-selection (with success history)

        :param population:      List of :class:`~modea.Individual.FloatIndividual` objects containing the previous generation
        :param new_population:  List of :class:`~modea.Individual.FloatIndividual` objects containing the new generation
        :param t:               Timestamp of the current generation being evaluated
        :param param:           :class:`~modea.Parameters.Parameters` object for storing all parameters, options, etc.
        :returns:               A slice of the sorted new_population list.
    """

    new_individual = new_population[0]
    individual = population[0]

    if new_individual.fitness < individual.fitness:
        param.best_fitness = new_individual.fitness
        result = new_population
        param.addToSuccessHistory(t, True)
    else:
        result = population
        param.addToSuccessHistory(t, False)

    return result
