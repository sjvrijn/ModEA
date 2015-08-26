#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import matplotlib.pyplot as plt
from code.Algorithms import onePlusOneES, CMSA_ES, CMA_ES, onePlusOneCholeskyCMAES, onePlusOneActiveCMAES

# Constant fitness function
def constantFitness(_):
    return 0.5

# Random fitness function
def randomFitness(_):
    return np.random.random(1)

# Sum fitness (minimize all parameters)
def sumFitness(individual):
    return np.sum(individual.dna)

# Sphere fitness function
def sphereFitness(individual):
    return np.sqrt(np.sum(np.square(individual.dna)))

# Rastrigin fitness function
def rastriginFitness(individual):
    return np.sum(np.square(individual.dna) + 10*np.cos(2*np.pi*individual.dna) + 10)


fitnes_functions = {'const': constantFitness, 'random': randomFitness, 'sum': sumFitness,
                    'sphere': sphereFitness, 'rastrigin': rastriginFitness, }
algorithms = {'1+1': onePlusOneES, 'CMSA': CMSA_ES, 'CMA': CMA_ES, 'Cholesky': onePlusOneCholeskyCMAES,
              'Active': onePlusOneActiveCMAES}


def run_tests():

    np.set_printoptions(linewidth=200)

    # Set parameters
    n = 10
    budget = 1000
    num_runs = 3
    fitnesses_to_test = ['sphere', 'rastrigin']  # ['const', 'random', 'sum', 'sphere', 'rastrigin']

    algorithms_to_test = ['1+1', 'CMA', 'CMSA', 'Cholesky', 'Active']  # ['1+1', 'CMA', 'CMSA', 'Cholesky', 'Active']

    # 'Catch' results
    results = {}
    sigmas = {}
    fitnesses = {}

    # Run algorithms
    for i, alg_name in enumerate(algorithms_to_test):

        print(alg_name)
        algorithm = algorithms[alg_name]
        results[alg_name] = {}
        sigmas[alg_name] = {}
        fitnesses[alg_name] = {}

        for fit_name in fitnesses_to_test:

            print("  {}".format(fit_name))
            results[alg_name][fit_name] = []
            sigmas[alg_name][fit_name] = None
            fitnesses[alg_name][fit_name] = None

            # Perform the actual run of the algorithm
            for _ in range(num_runs):
                results[alg_name][fit_name].append(algorithm(n, fitnes_functions[fit_name], budget))

            # Preprocess/unpack results
            _, sigmas[alg_name][fit_name], fitnesses[alg_name][fit_name] = (list(x) for x in zip(*results[alg_name][fit_name]))
            sigmas[alg_name][fit_name] = np.mean(np.array(sigmas[alg_name][fit_name]), axis=0)
            fitnesses[alg_name][fit_name] = np.mean(np.array(fitnesses[alg_name][fit_name]), axis=0)

    makeGraphs(sigmas, fitnesses, algorithms_to_test, fitnesses_to_test)

    print('Done!')


def makeGraphs(sigmas, fitnesses, alg_names, fit_names):

    fig = plt.figure(figsize=(20, 15))
    num_rows = len(alg_names)  # One row per algorithm
    num_colums = 2  # Fitness and Sigma

    for i, alg_name in enumerate(alg_names):

        # Plot results for this algorithm
        x_range = np.array(range(len(sigmas[alg_name][fit_names[0]])))

        sigma_plot = fig.add_subplot(num_rows, num_colums, num_colums*i + 1)
        sigma_plot.set_title('Sigma')
        fitness_plot = fig.add_subplot(num_rows, num_colums, num_colums*i + 2)
        fitness_plot.set_title('Fitness')

        for fitness_name in fit_names:
            sigma_plot.plot(x_range, sigmas[alg_name][fitness_name], label=fitness_name)
            fitness_plot.plot(x_range, fitnesses[alg_name][fitness_name], label=fitness_name)

        sigma_plot.legend(loc=0, fontsize='small')
        sigma_plot.set_title("Sigma over time ({})".format(alg_name))
        sigma_plot.set_xlabel('Evaluations')
        sigma_plot.set_ylabel('Sigma')
        sigma_plot.set_yscale('log')

        fitness_plot.legend(loc=0, fontsize='small')
        fitness_plot.set_title("Fitness over time ({})".format(alg_name))
        fitness_plot.set_xlabel('Evaluations')
        fitness_plot.set_ylabel('Fitness value')
        fitness_plot.set_yscale('log')

    fig.tight_layout()
    fig.savefig('../results_per_algorithm.png')


if __name__ == '__main__':
    run_tests()

    # from bbob.bbob_pproc import cococommands
    # help(cococommands)