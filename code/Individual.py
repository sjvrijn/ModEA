#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import copy
import numpy as np

class Individual(object):
    """
        Data holder class for ES individuals.
    """

    def __init__(self, n):
        """
            Create an Individual. Stores the DNA column vector and all individual-specific parameters
            :param n: dimensionality of the problem to be solved
        """
        self.n = n
        self.dna = np.random.randn(n,1)  # Column vector
        self.fitness = None  # Default 'unset' value
        self.sigma = 1

        self.last_z = np.zeros((n,1))
        self.mutation_vector = np.zeros((n,1))


    def getCopy(self):
        """
            Return a new Individual object that is a copy of the current copy
        """
        return_copy = Individual(self.n)
        return_copy.dna = copy.copy(self.dna)
        return_copy.fitness = self.fitness
        return_copy.sigma = self.sigma

        return_copy.last_z = self.last_z
        return_copy.mutation_vector = self.mutation_vector

        return return_copy