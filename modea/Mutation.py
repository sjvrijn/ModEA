#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This Module contains a collection of Mutation operators to be used in the ES-Framework

A Mutation operator mutates an Individual's genotype inline, thus returning nothing.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import random
from numpy import add, bitwise_and, dot, exp, floor, mod, shape, zeros
from numpy.linalg import norm
from random import gauss
from math import sqrt


'''-----------------------------------------------------------------------------
#                          Mutation Helper Functions                           #
-----------------------------------------------------------------------------'''


def _keepInBounds(x, l_bound, u_bound):
    """
        This function transforms x to t w.r.t. the low and high
        boundaries lb and ub. It implements the function T^{r}_{[a,b]} as
        described in Rui Li's PhD thesis "Mixed-Integer Evolution Strategies
        for Parameter Optimization and Their Applications to Medical Image
        Analysis" as alorithm 6.

        :param x:       Column vector to be kept in bounds
        :param l_bound: Lower bound column vector
        :param u_bound: Upper bound column vector
        :returns:       An in-bounds kept version of the column vector ``x``
    """

    y = (x - l_bound) / (u_bound - l_bound)
    floor_y = floor(y)                              # Local storage to prevent double calls
    I = mod(floor_y, 2) == 0
    yprime = zeros(shape(y))
    yprime[I] = np.abs(y[I] - floor_y[I])
    yprime[~I] = 1.0 - np.abs(y[~I] - floor_y[~I])

    x = l_bound + (u_bound - l_bound) * yprime
    return x


def adaptStepSize(individual):
    """
        Given the current individual, randomly determine a new step size offset
        that can be no greater than maxStepSize - baseStepSize

        :param individual:  The :class:`~modea.Individual.FloatIndividual` object whose step size should be adapted
    """
    # Empirically determined, see paper
    gamma = 0.22

    offset = individual.stepSizeOffset
    offset = 1 + ((1 - offset) / offset)
    offset = 1 / (offset * exp(gamma * gauss(0, 1)))
    individual.stepSizeOffset = min(offset, (individual.maxStepSize - individual.baseStepSize))


def _scaleWithThreshold(mutation_vector, threshold):
    """
        Checks if norm(mutation_vector) is at least the given threshold.
        If not, the vector is mirrored to the other side of the threshold,
        i.e. scaled to be length: threshold + (threshold - norm(mutation_vector))

        :param mutation_vector:  Mutation vector to be scaled
        :param threshold:        Minimum length threshold. Vector is scaled if length does not reach threshold
        :returns:                The threshold-compliant mutation vector
    """

    length = norm(mutation_vector)
    if length < threshold:
        new_length = threshold + (threshold - length)
        mutation_vector *= (new_length / length)

    return mutation_vector


def _adaptSigma(sigma, p_s, c=0.817):
    """
        Adapt parameter sigma based on the 1/5th success rule

        :param sigma:  Sigma value to be adapted
        :param p_s:    Recent success rate, determines whether sigma is increased or decreased
        :param c:      Factor c that is used to increase or decrease sigma
        :returns:      New value sigma
    """

    if p_s < 1/5:
        sigma *= c
    elif p_s > 1/5:
        sigma /= c

    return sigma


def _getXi():
    """
        Randomly returns 5/7 or 7/5 with equal probability
        :return: float Xi
    """
    if bool(random.getrandbits(1)):
        return 5/7
    else:
        return 7/5


'''-----------------------------------------------------------------------------
#                                ES Mutations                                  #
-----------------------------------------------------------------------------'''


def addRandomOffset(individual, param, sampler):
    """
        Mutation 1: x = x + sigma*N(0,I)

        :param individual:  :class:`~modea.Individual.FloatIndividual` to be mutated
        :param param:       :class:`~modea.Parameters.Parameters` object to store settings
        :param sampler:     :mod:`~modea.Sampling` module from which the random values should be drawn
    """
    individual.genotype += param.sigma * sampler.next()


def CMAMutation(individual, param, sampler, threshold_convergence=False):
    """
        CMA mutation: x = x + (sigma * B*D*N(0,I))

        :param individual:              :class:`~modea.Individual.FloatIndividual` to be mutated
        :param param:                   :class:`~modea.Parameters.Parameters` object to store settings
        :param sampler:                 :mod:`~modea.Sampling` module from which the random values should be drawn
        :param threshold_convergence:   Boolean: Should threshold convergence be applied. Default: False
    """

    individual.last_z = sampler.next()

    if threshold_convergence:
        individual.last_z = _scaleWithThreshold(individual.last_z, param.threshold)

    individual.mutation_vector = dot(param.B, (param.D * individual.last_z))  # y_k in cmatutorial.pdf)
    mutation_vector = individual.mutation_vector * param.sigma

    individual.genotype = _keepInBounds(add(individual.genotype, mutation_vector), param.l_bound, param.u_bound)


'''-----------------------------------------------------------------------------
#                                GA Mutations                                  #
-----------------------------------------------------------------------------'''


def mutateBitstring(individual):
    """
        Simple 1/n bit-flip mutation

        :param individual:  :mod:`~modea.Individual` with a bit-string as genotype to undergo p=1/n mutation
    """
    bitstring = individual.genotype
    n = len(bitstring)
    p = 1/n
    for i in range(n):
        if np.random.random() < p:
            bitstring[i] = 1-bitstring[i]


def mutateIntList(individual, param, num_options_per_module):
    """
        Self-adaptive random integer mutation to mutate the structure of an ES

        :param individual:              :class:`~modea.Individual.MixedIntegerIndividual` whose integer-part will be mutated
        :param param:                   :class:`~modea.Parameters.Parameters` object
        :param num_options_per_module:  List :data:`~modea.num_options` with the number of available modules per module
                                        position that are available to choose from
    """

    p = individual.baseStepSize + individual.stepSizeOffset
    num_ints = individual.num_ints

    int_list = individual.genotype[:num_ints-1]  # Get the relevant slice
    for i, val in enumerate(num_options_per_module):
        if np.random.random() < p:
            # -1 as random_integers is [1, val], -1 to simulate leaving out the current value
            new_int = np.random.random_integers(val-1)-1
            if int_list[i] == new_int:
                new_int = val - 1  # If we randomly selected the same value, pick the value we left out

            int_list[i] = new_int

    if np.random.random() < p:
        new_lambda = np.random.random_integers(param.l_bound[num_ints-1], param.u_bound[num_ints-1])
        individual.genotype[num_ints-1] = new_lambda


def mutateFloatList(individual, param, options):
    """
        Self-adaptive, uniformly random floating point mutation on the tunable parameters of an ES

        :param individual:  :class:`~modea.Individual.MixedIntegerIndividual` whose integer-part will be mutated
        :param param:       :class:`~modea.Parameters.Parameters` object
        :param options:     List of tuples :data:`~modea.options` with the number of tunable parameters per module
    """

    # Setup of values
    p = individual.baseStepSize + individual.stepSizeOffset
    float_part = individual.genotype[individual.num_ints:]
    int_part = individual.genotype[:individual.num_ints-1]
    l_bound = param.l_bound[individual.num_ints:].flatten()
    u_bound = param.u_bound[individual.num_ints:].flatten()
    search_space = u_bound - l_bound

    # Create the mask: which float values will actually be mutated?
    cond_mask = [True,True,True,True,True,True,True]  # TODO FIXME: these are default CMA parameters, make this dynamic!
    for i, val in enumerate(options):
        cond_mask.extend([bool(int_part[i] * 1)] * val[2])
    mutate_mask =np.random.random_sample(float_part.shape) < p
    combined_mask = bitwise_and(cond_mask, mutate_mask)

    # Scale the random values to the search space, then start at the lower bound
    float_part[combined_mask] =np.random.random_sample(float_part.shape)[combined_mask] * search_space[combined_mask]
    float_part[combined_mask] += l_bound[combined_mask]


def mutateMixedInteger(individual, param, options, num_options_per_module):
    """
        Self-adaptive mixed-integer mutation of the structure of an ES

        :param individual:              :class:`~modea.Individual.MixedIntegerIndividual` whose integer-part will be mutated
        :param param:                   :class:`~modea.Parameters.Parameters` object
        :param options:                 List of tuples :data:`~modea.options` with the number of tunable parameters per module
        :param num_options_per_module:  List :data:`~modea.num_options` with the number of available modules per module position
                                        that are available to choose from
    """
    adaptStepSize(individual)
    mutateIntList(individual, param, num_options_per_module)
    mutateFloatList(individual, param, options)


'''-----------------------------------------------------------------------------
#                               MIES Mutations                                 #
-----------------------------------------------------------------------------'''


def MIES_MutateDiscrete(individual, begin, end, u, num_options, options):
    """
        Mutate the discrete part of a Mixed-Integer representation

        :param individual:      The individual to mutate
        :param begin:           Start index of the discrete part of the individual's representation
        :param end:             End index of the discrete part of the individual's representation
        :param u:               A pre-determined random value from a Gaussian distribution
        :param num_options:     List :data:`~modea.num_options` with the number of available modules per module position
                                that are available to choose from
        :param options:         List of tuples :data:`~modea.options` with the number of tunable parameters per module
        :return:                A boolean mask array to be used for further conditional mutations based on which modules
                                are active
    """
    conditional_mask = [True,True,True,True,True,True,True]
    for x in range(begin, end):
        if individual.genotype[x] is not None:

            # set stepsize
            tau = 1 / sqrt(2 * individual.num_discrete)
            tau_prime = 1 / sqrt(2 * sqrt(individual.num_discrete))
            individual.stepSizeOffsetMIES[x] = 1 / (
                1 + ((1 - individual.stepSizeOffsetMIES[x]) / individual.stepSizeOffsetMIES[x]) * exp(
                    (-tau) * u - tau_prime * gauss(0.5, 1)))
            # Keep stepsize within the bounds
            baseMIESstep = 1 / (3 * num_options[x])  # p'_i = T[ 1 / (3n_d) , 0.5]
            individual.stepSizeOffsetMIES[x] = _keepInBounds(individual.stepSizeOffsetMIES[x], baseMIESstep, 0.5)

            threshold = np.random.random_sample()
            # change discrete
            if threshold < individual.stepSizeMIES(x):
                temparray = []
                for i in range(num_options[x]):
                    temparray.append(i)
                temparray.remove(individual.genotype[x])
                individual.genotype[x] = random.choice(temparray)
            for i in range(options[x][2]):
                conditional_mask.append(individual.genotype[x])

    return conditional_mask


def MIES_MutateIntegers(individual, begin, end, u, param):
    """
        Mutate the integer part of a Mixed-Integer representation

        :param individual:      The individual to mutate
        :param begin:           Start index of the integer part of the individual's representation
        :param end:             End index of the integer part of the individual's representation
        :param u:               A pre-determined random value from a Gaussian distribution
        :param param:           :class:`~modea.Parameters.Parameters` object
    """
    for x in range(begin, end):
        if individual.genotype[x] is not None:
            # Adapt stepsize
            tau = 1 / sqrt(2 * individual.num_ints)
            tau_prime = 1 / sqrt(2 * sqrt(individual.num_ints))
            individual.stepSizeOffsetMIES[x] = max(1,
                                                   individual.stepSizeOffsetMIES[x] * exp(tau * u + tau_prime * gauss(0.5, 1)))

            # Mudate integer
            psi = 1 - (individual.stepSizeMIES(x) / individual.num_ints) / (
                1 + sqrt(1 + pow(individual.stepSizeMIES(x) / individual.num_ints, 2)))
            u1, u2 = np.random.random_sample(2)
            G1 = int(floor(np.log(1 - u1) / np.log(1 - psi)))
            G2 = int(floor(np.log(1 - u2) / np.log(1 - psi)))
            individual.genotype[x] = individual.genotype[x] + G1 - G2
            # Keep the change within the bounds
            individual.genotype[x] = int(_keepInBounds(individual.genotype[x], param.l_bound[x], param.u_bound[x]))


def MIES_MutateFloats(conditional_mask, individual, begin, end, u, param):
    """
        Mutate the floating point part of a Mixed-Integer representation

        :param conditional_mask:    Conditional mask that determines which floating point values are allowed to mutate
        :param individual:          The individual to mutate
        :param begin:               Start index of the integer part of the individual's representation
        :param end:                 End index of the integer part of the individual's representation
        :param u:                   A pre-determined random value from a Gaussian distribution
        :param param:               :class:`~modea.Parameters.Parameters` object
    """
    for x in range(begin, end):
        if individual.genotype[x] is not None and conditional_mask[x-(individual.num_discrete+individual.num_ints)]:
            # Adapt stepsize
            tau = 1 / sqrt(2 * individual.num_floats)
            tau_prime = 1 / sqrt(2 * sqrt(individual.num_floats))
            individual.stepSizeOffsetMIES[x] = individual.stepSizeOffsetMIES[x] * exp(u * tau + gauss(0.5, 1) * tau_prime)

            # Mutate float
            rand = np.random.randn()
            old_value = individual.genotype[x]
            individual.genotype[x] += individual.stepSizeMIES(x) * rand
            individual.genotype[x] = _keepInBounds(individual.genotype[x], param.l_bound[x], param.u_bound[x])

            # Reverse-engineer the actual stepsize based on the final mutation step
            actual_stepsize = abs((individual.genotype[x] - old_value) / rand)
            individual.stepSizeOffsetMIES[x] = actual_stepsize - individual.baseStepSize


def MIES_Mutate(individual, param, options, num_options):
    """
        Self-adaptive mixed-integer mutation of the structure of an ES

        :param individual:  :class:`~modea.Individual.MixedIntegerIndividual` whose integer-part will be mutated
        :param param:       :class:`~modea.Parameters.Parameters` object
        :param options:     List of tuples :data:`~modea.options` with the number of tunable parameters per module
        :param num_options: List :data:`~modea.num_options` with the number of available modules per module position
                            that are available to choose from
    """

    u = gauss(0.5, 1)

    conditional_mask = MIES_MutateDiscrete(individual, 0, individual.num_discrete, u, num_options, options)
    MIES_MutateIntegers(individual, individual.num_discrete, individual.num_discrete+individual.num_ints, u, param)
    MIES_MutateFloats(conditional_mask, individual, individual.num_discrete+individual.num_ints, individual.n, u, param)
