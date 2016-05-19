#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

from copy import copy
import numpy as np

class ESIndividual(object):
    """
        Data holder class for ES individuals.
    """

    def __init__(self, n):
        """
            Create an ESIndividual. Stores the DNA column vector and all individual-specific parameters.
            Default DNA consists of np.ones((n,1))

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
            Return a new ESIndividual object that is a copy of the current object. Can be called using
            >>> import copy
            >>> copy.copy(ESIndividual())

            :returns:  ESIndividual object with all attributes explicitly copied
        """
        return_copy = ESIndividual(self.n)
        return_copy.genotype = copy(self.genotype)
        return_copy.fitness = self.fitness
        return_copy.sigma = self.sigma

        return_copy.last_z = copy(self.last_z)
        return_copy.mutation_vector = copy(self.mutation_vector)

        return return_copy


class GAIndividual(object):
    """
        Data holder class for GA individuals.
    """

    def __init__(self, n):
        """
            Create a GAIndividual. Stores the DNA column vector and all individual-specific parameters.
            Default DNA consists of np.ones((n,1))

            :param n: dimensionality of the problem to be solved
        """
        self.n = n
        self.genotype = np.ones((n, 1))               # Column vector
        self.fitness = np.inf                   # Default 'unset' value

        self.maxStepSize = 0.5
        self.initStepSize = 0.2
        self.sigma = 1

        # TODO: Remove these parameters from this class
        self.last_z = np.zeros((n,1))
        self.mutation_vector = np.zeros((n,1))

        if n > 5:
            self.baseStepSize = 1 / n
        else:
            self.baseStepSize = 0.175  # Random guess value, may need to be updated
        # The actual stepSize is the base + offset, so final starting stepSize = initStepSize
        self.stepSizeOffset = self.initStepSize - self.baseStepSize


    def __copy__(self):
        """
            Return a new Individual object that is a copy of the current object. Can be called using
            >>> import copy
            >>> copy.copy(GAIndividual())

            :returns:  Individual object with all attributes explicitly copied
        """
        return_copy = GAIndividual(self.n)
        return_copy.genotype = copy(self.genotype)
        return_copy.fitness = self.fitness
        return_copy.sigma = self.sigma

        return_copy.maxStepSize = self.maxStepSize
        return_copy.baseStepSize = self.baseStepSize
        return_copy.initStepSize = self.initStepSize
        return_copy.stepSizeOffset = self.stepSizeOffset

        return return_copy
