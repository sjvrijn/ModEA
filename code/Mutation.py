#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

"""
This Module contains a collection of Mutation operators to be used in the ES-Framework

A Mutation operator mutates an Individual's DNA inline, thus returning nothing.
"""


# TODO: Split (CMA-based) mutations into multiple/as many parts as possible.
# E.g. step size control & CMA

import numpy as np
from numpy import add, dot, exp
from numpy.linalg import norm
from random import gauss, getrandbits


def adaptStepSize(individual):
    """
        Given the current step size for a candidate, randomly determine a new step size offset,
        that can be no greater than maxStepSize - baseStepSize

        :param individual:  The Individual object whose step size should be adapted
    """
    # Empirically determined, see paper
    gamma = 0.22

    offset = individual.stepSizeOffset
    offset = 1 + ((1 - offset) / offset)
    offset = 1 / (offset * exp(gamma * gauss(0, 1)))
    individual.stepSizeOffset = min(offset, (individual.maxStepSize - individual.baseStepSize))


# TODO: come up with a better name for this mutation function
def x1(individual, param, sampler):
    """
        Mutation 1: x = x + sigma*N(0,I)

        :param individual:  Individual to be mutated
        :param param:       Parameters object to store settings
        :param sampler:     Sampler from which the random values should be drawn
    """

    individual.dna += param.sigma * sampler.next()


def CMAMutation(individual, param, sampler, threshold_convergence=False):
    """
        CMA mutation: x = x + (sigma * B*D*N(0,I))

        :param individual:              Individual to be mutated
        :param param:                   Parameters object to store settings
        :param sampler:                 Sampler from which the random values should be drawn
        :param threshold_convergence:   Boolean: Should threshold convergence be applied. Default: False
    """

    individual.last_z = sampler.next()

    if threshold_convergence:
        individual.last_z = _scaleWithThreshold(individual.last_z, param.threshold)

    individual.mutation_vector = dot(param.B, (param.D * individual.last_z))  # y_k in cmatutorial.pdf)
    mutation_vector = individual.mutation_vector * param.sigma

    individual.dna = add(individual.dna, mutation_vector)


def choleskyCMAMutation(individual, param, sampler):
    """
        Cholesky CMA based mutation

        :param individual:  Individual to be mutated
        :param param:       Parameters object to store settings
        :param sampler:     Sampler from which the random values should be drawn
    """

    param.last_z = sampler.next()
    mutation_vector = np.dot(param.A, param.last_z.T)

    individual.dna += param.sigma * mutation_vector


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
    if bool(getrandbits(1)):
        return 5/7
    else:
        return 7/5


### GA MUTATIONS ###
def mutateBitstring(individual):
    """ Extremely simple 1/n bit-flip mutation """
    bitstring = individual.dna
    n = len(bitstring)
    p = 1/n
    for i in range(n):
        if np.random.random() < p:
            bitstring[i] = 1-bitstring[i]


def mutateIntList(individual, num_options):
    """ self-adaptive random integer mutation """

    adaptStepSize(individual)
    p = individual.baseStepSize + individual.stepSizeOffset

    int_list = individual.dna
    for i in range(len(individual.dna)):
        if np.random.random() < p:
            # -1 as random_integers is [1, val], -1 to simulate leaving out the current value
            new_int = np.random.random_integers(num_options[i]-1)-1
            if int_list[i] == new_int:
                new_int = num_options[i] - 1  # If we randomly selected the same value, pick the value we left out

            int_list[i] = new_int
