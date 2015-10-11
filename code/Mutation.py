#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sander van Rijn <svr003@gmail.com>'

"""
This Module contains a collection of Mutation operators to be used in the ES-Framework

A Mutation operator mutates an Individual's DNA inline, thus returning nothing.
"""

# TODO: Update all functions & usage to allow for (e.g.) mirrored sampling; i.e. returning multiple samples per given
# TODO: individual

# TODO: Split (CMA-based) mutations into multiple/as many parts as possible. E.g. step size control & CMA

import numpy as np
from numpy import add, dot, exp
from numpy.linalg import norm
from numpy.random import randn
from random import getrandbits


# TODO: come up with a better name for this mutation function
def x1(individual, parameters, sampler):
    """ Mutation 1: x = x + sigma*N(0,I) """

    n = individual.n
    individual.dna += parameters.sigma * sampler.next()


def CMAMutation(individual, parameters, sampler):
    """ CMA based mutation: x = x + ((sigma_mean*tau*N(0,1)) * (B*D*N(0,I))) """

    n = individual.n
    individual.sigma = parameters.sigma_mean * exp(parameters.tau * randn(1,1))
    individual.mutation_vector = dot(parameters.B, (parameters.D * sampler.next()))  # B*D*randn(n,1)
    individual.last_z = individual.sigma * individual.mutation_vector

    individual.dna += individual.last_z


# TODO FIXME: This should probably be the actual base function
def CMAMutation__(individual, parameters, sampler, threshold_convergence=False):
    """ CMA mutation: x = x + (sigma * B*D*N(0,I)) """

    n = individual.n
    individual.last_z = sampler.next()
    individual.mutation_vector = dot(parameters.B, (parameters.D * individual.last_z))  # y_k in cmatutorial.pdf)

    mutation_vector = individual.mutation_vector * parameters.sigma
    if threshold_convergence:
        individual.mutation_vector = scaleWithThreshold(mutation_vector, parameters.threshold) / parameters.sigma

    individual.dna = add(individual.dna, mutation_vector)


def scaleWithThreshold(mutation_vector, threshold):
    """
        Checks if norm(mutation_vector) is at least the given threshold.
        If not, the vector is mirrored to the other side of the threshold,
        i.e. scaled to be length: threshold + (threshold - norm(mutation_vector))
    """

    length = norm(mutation_vector)
    if length < threshold:
        new_length = threshold + (threshold - length)
        mutation_vector *= (new_length / length)

    return mutation_vector

def choleskyCMAMutation(individual, parameters, sampler):
    """ Cholesky CMA based mutation """

    parameters.last_z = sampler.next()
    mutation_vector = np.dot(parameters.A, parameters.last_z.T)

    individual.dna += parameters.sigma * mutation_vector


def adaptSigma(sigma, p_s, c=0.817):
    """ Adapt parameter sigma based on the 1/5th success rule """

    if p_s < 1/5:
        sigma *= c
    elif p_s > 1/5:
        sigma /= c

    return sigma


def calculateRotationMatrix(rotations):
    """ Given a list of rotation matrices (R_ij), calculate the final matrix C """

    pass


def getXi():
    if bool(getrandbits(1)):
        return 5/7
    else:
        return 7/5
