#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import matplotlib.pyplot as plt
import sys
from bbob import bbobbenchmarks
from bbob import fgeneric
from code.Algorithms import onePlusOneES, CMSA_ES, CMA_ES, onePlusOneCholeskyCMAES, onePlusOneActiveCMAES

# Constant fitness function
def constantFitness(_):
    return 0.5

# Random fitness function
def randomFitness(_):
    return np.random.random(1)

# Sum fitness (minimize all parameters)
def sumFitness(individual):
    return np.sum(individual)

# Sphere fitness function
def sphereFitness(individual):
    return np.sqrt(np.sum(np.square(individual)))

# Rastrigin fitness function
def rastriginFitness(individual):
    return np.sum(np.square(individual) + 10*np.cos(2*np.pi*individual) + 10)


# ['const', 'random', 'sum', 'sphere', 'rastrigin']
my_fitness_functions = {'const': constantFitness, 'random': randomFitness, 'sum': sumFitness,
                        'sphere': sphereFitness, 'rastrigin': rastriginFitness, }

# Sets of noise-free and noisy benchmarks
free_function_ids = bbobbenchmarks.nfreeIDs
noisy_function_ids = bbobbenchmarks.noisyIDs

fitness_functions = {'sphere': free_function_ids[0], 'elipsoid': free_function_ids[1],
                     'rastrigin': free_function_ids[2], }
algorithms = {'1+1': onePlusOneES, 'CMSA': CMSA_ES, 'CMA': CMA_ES, 'Cholesky': onePlusOneCholeskyCMAES,
              'Active': onePlusOneActiveCMAES}

datapath = "test_results/"  # Where to store results
opts = {'algid': None,
        'comments': '<comments>',
        'inputformat': 'col'}  # 'row' or 'col'

def sysPrint(string):
    """ Small function to take care of the 'overhead' of sys.stdout.write + flush """
    sys.stdout.write(string)
    sys.stdout.flush()

def run_tests():
    """ Single function to run all desired combinations of algorithms * fitness functions """

    np.set_printoptions(linewidth=200)

    # Set parameters
    save_pdf = False
    n = 10
    budget = 1000
    num_runs = 5
    fitnesses_to_test = ['sphere', 'elipsoid', 'rastrigin']  # ['sphere', 'elipsoid', 'rastrigin']
    # fitnesses_to_test = ['sphere']

    # algorithms_to_test = ['1+1', 'CMA', 'CMSA', 'Cholesky', 'Active']  # ['1+1', 'CMA', 'CMSA', 'Cholesky', 'Active']
    algorithms_to_test = ['CMA']

    # 'Catch' results
    results = {}
    targets = {}
    sigmas = {}
    fitnesses = {}
    avg_sigmas = {}
    avg_fitnesses = {}


    # Run algorithms
    for i, alg_name in enumerate(algorithms_to_test):

        opts['algid'] = alg_name
        f = fgeneric.LoggingFunction(datapath, **opts)

        print(alg_name)
        algorithm = algorithms[alg_name]
        results[alg_name] = {}
        targets[alg_name] = {}
        sigmas[alg_name] = {}
        fitnesses[alg_name] = {}
        avg_sigmas[alg_name] = {}
        avg_fitnesses[alg_name] = {}

        for fit_name in fitnesses_to_test:

            fun_id = fitness_functions[fit_name]

            print('  {}'.format(fit_name))
            results[alg_name][fit_name] = []
            targets[alg_name][fit_name] = []
            sigmas[alg_name][fit_name] = None
            fitnesses[alg_name][fit_name] = None

            # Perform the actual run of the algorithm
            for j in range(num_runs):
                sysPrint('    Run: {}\r'.format(j))  # I want the actual carriage return here! No output clutter
                f_target = f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=j)).ftarget
                targets[alg_name][fit_name].append(f_target)
                results[alg_name][fit_name].append(algorithm(n, f.evalfun, budget))

            # Preprocess/unpack results
            _, sigmas[alg_name][fit_name], fitnesses[alg_name][fit_name] = (list(x) for x in zip(*results[alg_name][fit_name]))
            sigmas[alg_name][fit_name] = np.array(sigmas[alg_name][fit_name]).T
            avg_sigmas[alg_name][fit_name] = np.mean(sigmas[alg_name][fit_name], axis=1)

            fitnesses[alg_name][fit_name] = np.array(fitnesses[alg_name][fit_name]).T
            fitnesses[alg_name][fit_name] = np.subtract(fitnesses[alg_name][fit_name], np.array(targets[alg_name][fit_name])[np.newaxis,:])
            avg_fitnesses[alg_name][fit_name] = np.mean(fitnesses[alg_name][fit_name], axis=1)

    sysPrint('Creating graphs.')

    makeGraphsPerAlgorithm(sigmas, fitnesses, algorithms_to_test, fitnesses_to_test, save_pdf=save_pdf)
    sysPrint('.')
    makeGraphsPerFitness(sigmas, fitnesses, algorithms_to_test, fitnesses_to_test, save_pdf=save_pdf)
    sysPrint('.')
    makeGraphsPerAlgorithm(avg_sigmas, avg_fitnesses, algorithms_to_test, fitnesses_to_test, suffix='_avg', save_pdf=save_pdf)
    print('.')
    makeGraphsPerFitness(avg_sigmas, avg_fitnesses, algorithms_to_test, fitnesses_to_test, suffix='_avg', save_pdf=save_pdf)

    print('Done!')


def makeGraphsPerAlgorithm(sigmas, fitnesses, alg_names, fit_names, suffix='', save_pdf=False):

    if save_pdf:
        extension = 'pdf'
    else:
        extension = 'png'

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
        for fit_name in fit_names:
            sigma_plot.plot(x_range, sigmas[alg_name][fit_name], label=fit_name)
            fitness_plot.plot(x_range, fitnesses[alg_name][fit_name], label=fit_name)

        sigma_plot.legend(loc=0, fontsize='small')
        sigma_plot.set_title("Sigma over time ({})".format(alg_name))
        sigma_plot.set_xlabel('Generations')
        sigma_plot.set_ylabel('Sigma')
        sigma_plot.set_yscale('log')

        fitness_plot.legend(loc=0, fontsize='small')
        fitness_plot.set_title("Fitness over time ({})".format(alg_name))
        fitness_plot.set_xlabel('Generations')
        fitness_plot.set_ylabel('Fitness value')
        fitness_plot.set_yscale('log')

    fig.tight_layout()
    fig.savefig('../results_per_algorithm{}.{}'.format(suffix, extension))


#TODO: make length of results-arrays equal for all algorithms. that is extend/lengthen by num_evals/generation
def makeGraphsPerFitness(sigmas, fitnesses, alg_names, fit_names, suffix='', save_pdf=False):

    if save_pdf:
        extension = 'pdf'
    else:
        extension = 'png'

    fig = plt.figure(figsize=(20, 15))
    num_rows = len(fit_names)  # One row per fitness function
    num_colums = 2  # Fitness and Sigma

    for i, fit_name in enumerate(fit_names):

        sigma_plot = fig.add_subplot(num_rows, num_colums, num_colums*i + 1)
        sigma_plot.set_title('Sigma')
        fitness_plot = fig.add_subplot(num_rows, num_colums, num_colums*i + 2)
        fitness_plot.set_title('Fitness')

        for alg_name in alg_names:
            # Plot results for this algorithm
            x_range = np.array(range(len(sigmas[alg_name][fit_names[0]])))

            sigma_plot.plot(x_range, sigmas[alg_name][fit_name], label=alg_name)
            fitness_plot.plot(x_range, fitnesses[alg_name][fit_name], label=alg_name)

        sigma_plot.legend(loc=0, fontsize='small')
        sigma_plot.set_title("Sigma over time ({})".format(fit_name))
        sigma_plot.set_xlabel('Generations')
        sigma_plot.set_ylabel('Sigma')
        sigma_plot.set_yscale('log')

        fitness_plot.legend(loc=0, fontsize='small')
        fitness_plot.set_title("Fitness over time ({})".format(fit_name))
        fitness_plot.set_xlabel('Generations')
        fitness_plot.set_ylabel('Fitness value')
        fitness_plot.set_yscale('log')

    fig.tight_layout()
    fig.savefig('../results_per_fitness{}.{}'.format(suffix, extension))



if __name__ == '__main__':
    np.random.seed(42)
    run_tests()

    # from bbob.bbob_pproc import cococommands
    # help(cococommands)
