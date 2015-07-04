__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
from random import getrandbits


def N(x, C):
    """
        Generate a vector sampled according to the normal distribution with the given parameters
    """

    pass


def x1(individual, sigma):
    """
        Mutation 1: x = x + sigma*N(0,I)
    """

    n = individual.n
    individual.dna += sigma * (np.random.random(n).reshape((n,1)) - 0.5)


def x2(x, parameter):
    # def adaptBaseWithParameter(base, parameter):
    """
        Mutation 2: x = x + tensorProduct(parameter, N(0,I))
    """

    # I = identityMatrix(n)

    # new_x = x + tensorProduct(parameter, N(0, I))
    # return new_x
    pass


def adaptSigma(sigma, p_s, c=0.817):
    """
        Adapt parameter sigma based on the 1/5th success rule
    """

    if p_s < 1/5:
        sigma *= c
    elif p_s > 1/5:
        sigma /= c

    return sigma


def calculateP_S(num_successes, num_failures):
    """
        Calculate the recent success rate
    """

    return num_successes / (num_failures + num_successes)


def calculateSigma_i(avgSigma, eta, tau):
    """
        Calculate separate sigma_i's for each i
    """

    # I = identityMatrix(n)

    # sigma_i = avgSigma * exp(eta + (tau * N(0, I)))
    pass


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