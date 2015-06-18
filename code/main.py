__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
from .Algorithms import onePlusOneES

# Constant fitness function
constantFitness = lambda x: 0.5

# Random fitness function
randomFitness = lambda x: np.random.random(1)


if __name__ == '__main__':

    n = 10
    budget = 1000
    fitnessFunction = constantFitness

    onePlusOneES(n, fitnessFunction, 1000)