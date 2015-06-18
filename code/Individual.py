__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np


class Individual(object):
    """
        Data holder class for ES individuals.
    """


    def __init__(self, n):
        self.n = n

        self.fitness = None  # Default 'unset' value