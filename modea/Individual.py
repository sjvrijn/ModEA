#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains definitions of Individual classes, that allow for a common interface with different
underlying genotypes (float, int, mixed-integer).
"""
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

from copy import copy
import numpy as np

class FloatIndividual(object):
    """
        Data holder class for individuals using a vector of floating point values as genotype.
        This type of individual can therefore be used by an Evolution Strategy (ES) such as the CMA-ES.
        Stores the genotype column vector and all individual-specific parameters.
        Default genotype is np.ones((n,1))

        :param n: dimensionality of the problem to be solved
    """

    def __init__(self, n):
        self.n = n
        self.genotype = np.ones((n, 1))         # Column vector
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

    def __repr__(self):
        return "<FloatIndividual [{}]: {}>".format(
            str(np.round_(self.fitness,2)),
            str(np.round_(self.genotype.flatten(), 2)),
        )

    def __str__(self):
        return self.__repr__()


class MixedIntIndividualError(Exception):
    pass

class MixedIntIndividual(object):
    """
        Data holder class for individuals using a vector containing both floating point and integer values as genotype.
        This type of individual can therefore be used by a GA with mixed-integer mutation operators.
        Stores the genotype column vector and all individual-specific parameters.
        Default genotype is np.ones((n,1))

        :param n:            Dimensionality of the problem to be solved, consisting of discrete, integers and
                             floating point values
        :param num_discrete: Number of discrete values in the genotype.
        :param num_ints:     Number of integer values in the genotype.
        :param num_floats:   Number of floating point values in the genotype.
    """

    def __init__(self, n, num_discrete, num_ints, num_floats=None):

        if n < 2:
            raise MixedIntIndividualError("Cannot define a mixed-integer representation in < 2 dimensions")
        if num_floats is None and num_discrete is None and num_ints is None:
            raise MixedIntIndividualError("Number of discrete, integer or floating point values not specified")

        self.n = n
        self.num_discrete = num_discrete
        self.num_ints = num_ints
        self.num_floats = n - (num_discrete + num_ints)
        self.genotype = np.ones((n, 1))                                       # Column vector
        self.fitness = np.inf                                                 # Default 'unset' value
        self.sigma = 1
        self.stepSizeOffsetMIES = np.ones(n)
        # Self-adaptive step size parameters
        self.maxStepSize = 0.5
        self.initStepSize = 0.2

        if n > 5:
            self.baseStepSize = 1 / n
        else:
            self.baseStepSize = 0.175  # Random guess value, may need to be updated

        for x in range(self.n):
            # self.baseStepSizeMIES[x] = 1/(3 * num_options[x])
            self.stepSizeOffsetMIES[x] = self.initStepSize - self.baseStepSize


    @property
    def stepsizeMIES(self):
        return self.stepSizeOffsetMIES + self.baseStepSize

    def __copy__(self):
        """
            Return a new Individual object that is a copy of the current object. Can be called using
            >>> import copy
            >>> copy.copy(MixedIntIndividual())

            :returns:  Individual object with all attributes explicitly copied
        """
        return_copy = MixedIntIndividual(self.n, self.num_discrete, self.num_ints)
        return_copy.genotype = copy(self.genotype)
        return_copy.fitness = self.fitness
        return_copy.sigma = self.sigma

        return_copy.maxStepSize = self.maxStepSize
        return_copy.baseStepSize = self.baseStepSize
        return_copy.initStepSize = self.initStepSize
        return_copy.stepSizeOffsetMIES=copy(self.stepSizeOffsetMIES)
        return return_copy
