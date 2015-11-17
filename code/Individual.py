#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sander van Rijn <svr003@gmail.com>'

from copy import copy
import numpy as np

class Individual(object):
    """
        Data holder class for ES individuals.
    """

    def __init__(self, n):
        """
            Create an Individual. Stores the DNA column vector and all individual-specific parameters.
            Default DNA consists of np.ones((n,1))

            :param n: dimensionality of the problem to be solved
        """
        self.n = n
        self.dna = np.ones((n,1))               # Column vector
        self.fitness = None                     # Default 'unset' value
        self.sigma = 1

        self.last_z = np.zeros((n,1))
        self.mutation_vector = np.zeros((n,1))


    def __copy__(self):
        """
            Return a new Individual object that is a copy of the current object. Can be called using
            >>> import copy
            >>> copy.copy(Individual())

            :returns:  Individual object with all attributes explicitly copied
        """
        return_copy = Individual(self.n)
        return_copy.dna = copy(self.dna)
        return_copy.fitness = self.fitness
        return_copy.sigma = self.sigma

        return_copy.last_z = copy(self.last_z)
        return_copy.mutation_vector = copy(self.mutation_vector)

        return return_copy
