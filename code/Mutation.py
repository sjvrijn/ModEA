#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from random import getrandbits


# TODO: come up with a better name for this mutation function
def x1(individual, parameters, sampler):
    """
        Mutation 1: x = x + sigma*N(0,I)

        :param individual:              Individual to be mutated
        :param parameters:              Parameters object to store settings
        :param sampler:                 Sampler from which the random values should be drawn
    """

    individual.dna += parameters.sigma * sampler.next()


def CMAMutation(individual, parameters, sampler, threshold_convergence=False):
    """
        CMA mutation: x = x + (sigma * B*D*N(0,I))

        :param individual:              Individual to be mutated
        :param parameters:              Parameters object to store settings
        :param sampler:                 Sampler from which the random values should be drawn
        :param threshold_convergence:   Boolean: Should threshold convergence be applied. Default: False
    """

    individual.last_z = sampler.next()

    if threshold_convergence:
        individual.last_z = _scaleWithThreshold(individual.last_z, parameters.threshold)

    individual.mutation_vector = dot(parameters.B, (parameters.D * individual.last_z))  # y_k in cmatutorial.pdf)
    mutation_vector = individual.mutation_vector * parameters.sigma

    # if threshold_convergence:
    #     individual.mutation_vector = _scaleWithThreshold(mutation_vector, parameters.threshold) / parameters.sigma

    individual.dna = add(individual.dna, mutation_vector)


def choleskyCMAMutation(individual, parameters, sampler):
    """
        Cholesky CMA based mutation

        :param individual:  Individual to be mutated
        :param parameters:  Parameters object to store settings
        :param sampler:     Sampler from which the random values should be drawn
    """

    parameters.last_z = sampler.next()
    mutation_vector = np.dot(parameters.A, parameters.last_z.T)

    individual.dna += parameters.sigma * mutation_vector


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
