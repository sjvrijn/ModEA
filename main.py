__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import matplotlib.pyplot as plt
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
    individual.fitness = np.sqrt(np.sum(np.square(individual.dna)))


fitnes_functions = {'const': constantFitness, 'random': randomFitness, 'sum': sumFitness, 'sphere': sphereFitness}


def run_tests():

    # Set parameters
    n = 10
    budget = 1000
    num_runs = 30
    # fitnesses_to_test = ['const', 'random', 'sum', 'sphere']
    fitnesses_to_test = ['const', 'sphere']

    # 'Catch' results
    results = {}
    sigmas = {}
    fitnesses = {}

    # Run algorithms
    for name in fitnesses_to_test:

        results[name] = []
        sigmas[name] = None
        fitnesses[name] = None

        for i in range(num_runs):
            results[name].append(onePlusOneES(n, fitnes_functions[name], budget))

        # Preprocess/unpack results
        _, sigmas[name], fitnesses[name] = (list(x) for x in zip(*results[name]))
        sigmas[name] = np.mean(np.array(sigmas[name]), axis=0)
        fitnesses[name] = np.mean(np.array(fitnesses[name]), axis=0)


    # Plot results
    x_range = np.array(range(budget))
    nil_line = np.zeros(budget)

    fig = plt.figure()

    sigma_plot = fig.add_subplot(1, 2, 1)
    sigma_plot.set_title('Sigma')
    fitness_plot = fig.add_subplot(1, 2, 2)
    fitness_plot.set_title('Fitness')

    for name in fitnesses_to_test:

        sigma_plot.plot(x_range, sigmas[name])
        sigma_plot.set_title("{}: sigma".format(name))

        fitness_plot.plot(x_range, fitnesses[name])
        fitness_plot.set_title("{}: fitness".format(name))

        # sub_plot = fig.add_subplot(len(fitnesses_to_test), 2, (2*i) + 1)
        # sub_plot.plot(x_range, sigmas[name])
        # sub_plot.plot(x_range, nil_line)
        # sub_plot.set_title("{}: sigma".format(name))
        #
        # sub_plot = fig.add_subplot(len(fitnesses_to_test), 2, (2*i) + 2)
        # sub_plot.plot(x_range, fitnesses[name])
        # sub_plot.plot(x_range, nil_line)
        # sub_plot.set_title("{}: fitness".format(name))

    fig.show()
    input("done")




if __name__ == '__main__':
    run_tests()