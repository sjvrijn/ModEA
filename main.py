__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
from code.Algorithms import onePlusOneES

# Constant fitness function
def constantFitness(individual):
    individual.fitness = 0.5

# Random fitness function
def randomFitness(individual):
    individual.fitness = np.random.random(1)

# Sum fitness (minimize all parameters)
def sumFitness(individual):
    individual.fitness = np.sum(individual.dna)

# Sphere fitness function
def sphereFitness(individual):
    individual.fitness = np.sum(np.square(individual.dna))


if __name__ == '__main__':

    n = 10
    budget = 1000

    onePlusOneES(n, constantFitness, 1000)
    onePlusOneES(n, randomFitness, 1000)