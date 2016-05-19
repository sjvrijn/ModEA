#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import matplotlib.pyplot as plt
import sys
from bbob import bbobbenchmarks
from bbob import fgeneric
from code import getOpts, options
from code.Algorithms import onePlusOneES, CMSA_ES, CMA_ES, onePlusOneCholeskyCMAES, onePlusOneActiveCMAES, customizedES

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
    algorithms_to_test = ['CMA']  # ['1+1', 'CMA', 'CMSA', 'Cholesky', 'Active']

    # 'Catch' results
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

        sigmas[alg_name] = {}
        fitnesses[alg_name] = {}
        avg_sigmas[alg_name] = {}
        avg_fitnesses[alg_name] = {}

        for fit_name in fitnesses_to_test:

            sigmas[alg_name][fit_name], fitnesses[alg_name][fit_name] = runAlgorithm(fit_name, algorithm, n, num_runs, f, budget)
            avg_sigmas[alg_name][fit_name] = np.mean(sigmas[alg_name][fit_name], axis=1)
            avg_fitnesses[alg_name][fit_name] = np.mean(fitnesses[alg_name][fit_name], axis=1)

    sysPrint('Creating graphs.')

    data = (sigmas, fitnesses, algorithms_to_test, fitnesses_to_test)
    makeGraphsPerAlgorithm(*data, save_pdf=save_pdf)
    sysPrint('.')
    makeGraphsPerFitness(*data, save_pdf=save_pdf)
    sysPrint('.')

    avg_data = (avg_sigmas, avg_fitnesses, algorithms_to_test, fitnesses_to_test)
    makeGraphsPerAlgorithm(*avg_data, suffix='_avg', save_pdf=save_pdf)
    print('.')
    makeGraphsPerFitness(*avg_data, suffix='_avg', save_pdf=save_pdf)

    print('Done!')


def runAlgorithm(fit_name, algorithm, n, num_runs, f, budget):

    fun_id = fitness_functions[fit_name]
    print('  {}'.format(fit_name))
    results = []
    targets = []

    # Perform the actual run of the algorithm
    for j in range(num_runs):
        sysPrint('    Run: {}\r'.format(j))  # I want the actual carriage return here! No output clutter
        f_target = f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=j)).ftarget
        targets.append(f_target)
        results.append(algorithm(n, f.evalfun, budget))

    # Preprocess/unpack results
    _, sigmas, fitnesses = (list(x) for x in zip(*results))
    sigmas = np.array(sigmas).T
    fitnesses = np.subtract(np.array(fitnesses).T, np.array(targets)[np.newaxis,:])

    return sigmas, fitnesses


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
    np.set_printoptions(linewidth=200)
    np.random.seed(42)
    run_tests()

    # from bbob.bbob_pproc import cococommands
    # help(cococommands)
