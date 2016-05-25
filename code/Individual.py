#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

from copy import copy
import numpy as np

class FloatIndividual(object):
    """
        Data holder class for individuals using a vector of floating point values as genotype.
        This type of individual can therefore be used by an Evolution Strategy (ES) such as the CMA-ES.
    """

    def __init__(self, n):
        """
            Create an FloatIndividual. Stores the genotype column vector and all individual-specific parameters.
            Default genotype is np.ones((n,1))

            :param n: dimensionality of the problem to be solved
        """
        self.n = n
        self.genotype = np.ones((n, 1))               # Column vector
        self.fitness = np.inf                   # Default 'unset' value

        self.sigma = 1

        self.last_z = np.zeros((n,1))
        self.mutation_vector = np.zeros((n,1))


    def __copy__(self):
        """
            Return a new FloatIndividual object that is a copy of the current object. Can be called using
            >>> import copy
            >>> copy.copy(FloatIndividual())

            :returns:  FloatIndividual object with all attributes explicitly copied
        """
        return_copy = FloatIndividual(self.n)
        return_copy.genotype = copy(self.genotype)
        return_copy.fitness = self.fitness
        return_copy.sigma = self.sigma

        return_copy.last_z = copy(self.last_z)
        return_copy.mutation_vector = copy(self.mutation_vector)

        return return_copy


class MixedIntIndividual(object):
    """
        Data holder class for individuals using a vector containing both floating point and integer values as genotype.
        This type of individual can therefore be used by an Evolution Strategy (ES) such as the CMA-ES.
    """

    def __init__(self, n):
        """
            Create a MixedIntIndividual. Stores the genotype column vector and all individual-specific parameters.
            Default genotype is np.ones((n,1))

            :param n: dimensionality of the problem to be solved
        """
        self.n = n
        self.genotype = [np.ones((n[0], 1)), np.ones((n[1], 1))]    # [Integer part, Real part] (column vectors)
        self.fitness = np.inf                                       # Default 'unset' value

        # Self-adaptive step size parameters
        self.maxStepSize = 0.5
        self.initStepSize = 0.2
        self.sigma = 1

        if np.sum(n) > 5:
            self.baseStepSize = 1 / np.sum(n)
        else:
            self.baseStepSize = 0.175  # Random guess value, may need to be updated
        # The actual stepSize is the base + offset, so final starting stepSize = initStepSize
        self.stepSizeOffset = self.initStepSize - self.baseStepSize


    def __copy__(self):
        """
            Return a new Individual object that is a copy of the current object. Can be called using
            >>> import copy
            >>> copy.copy(MixedIntIndividual())

            :returns:  Individual object with all attributes explicitly copied
        """
        return_copy = MixedIntIndividual(self.n)
        return_copy.genotype = copy(self.genotype)
        return_copy.fitness = self.fitness
        return_copy.sigma = self.sigma

        return_copy.maxStepSize = self.maxStepSize
        return_copy.baseStepSize = self.baseStepSize
        return_copy.initStepSize = self.initStepSize
        return_copy.stepSizeOffset = self.stepSizeOffset

        return return_copy
