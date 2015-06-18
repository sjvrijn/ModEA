__author__ = 'Sander van Rijn <svr003@gmail.com>'

import copy
import numpy as np

class Individual(object):
    """
        Data holder class for ES individuals.
    """

    def __init__(self, n):
        self.n = n
        self.dna = np.random.random(n)
        self.fitness = None  # Default 'unset' value


    def getCopy(self):
        """
            Return a new Individual object that is a copy of the current copy
        """
        return_copy = Individual(self.n)
        return_copy.dna = copy.copy(self.dna)
        return_copy.fitness = self.fitness

        return return_copy