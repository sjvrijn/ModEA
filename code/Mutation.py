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
from random import getrandbits


# TODO: come up with a better name for this mutation function
def x1(individual, parameters):
    """
        Mutation 1: x = x + sigma*N(0,I)
    """

    n = individual.n
    individual.dna += parameters.sigma * np.random.randn(n,1)


def CMAMutation(individual, parameters):
    """
        CMA based mutation: x = x + ((sigma_mean*tau*N(0,1)) * (B*D*N(0,I)))
    """

    n = individual.n
    individual.sigma = parameters.sigma_mean * np.exp(parameters.tau * np.random.randn(1,1))
    individual.last_s = np.dot(parameters.B, (parameters.D * np.random.randn(n,1)))  # B*D*randn(n,1)
    individual.last_z = individual.sigma * individual.last_s

    individual.dna += individual.last_z


def CMAMutation__(individual, parameters):  # TODO FIXME: This should probably be the actual base function
    """
        CMA mutation: x = x + (sigma * B*D*N(0,I))
    """

    n = individual.n
    individual.last_z = np.random.randn(n,1)
    # print(parameters.D, individual.last_z)
    individual.last_s = np.dot(parameters.B, (parameters.D * individual.last_z))  # Noted as y_k in cmatutorial.pdf)

    individual.dna += parameters.sigma * individual.last_s


def choleskyCMAMutation(individual, parameters):
    """
        Cholesky CMA based mutation
    """

    parameters.last_z = np.random.randn(1,individual.n)
    mutation_vector = parameters.sigma * np.dot(parameters.A, parameters.last_z.T)

    individual.dna += mutation_vector


def adaptSigma(sigma, p_s, c=0.817):
    """
        Adapt parameter sigma based on the 1/5th success rule
    """

    if p_s < 1/5:
        sigma *= c
    elif p_s > 1/5:
        sigma /= c

    return sigma


def calculateRotationMatrix(rotations):
    """
        Given a list of rotation matrices (R_ij), calculate the final matrix C
    """

    pass


def getXi():
    if bool(getrandbits(1)):
        return 5/7
    else:
        return 7/5