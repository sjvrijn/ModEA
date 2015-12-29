#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import sys
from bbob import bbobbenchmarks, fgeneric
from copy import copy
from code import getOpts, getBitString, options, num_options
from code.Algorithms import customizedES, baseAlgorithm
from code.Individual import Individual
from code.Parameters import Parameters
import code.Mutation as Mut
import code.Selection as Sel
import code.Recombination as Rec


# BBOB parameters: Sets of noise-free and noisy benchmarks
free_function_ids = bbobbenchmarks.nfreeIDs
noisy_function_ids = bbobbenchmarks.noisyIDs
datapath = "test_results/"  # Where to store results
# Options to be stored in the log file(s)
bbob_opts = {'algid': None,
             'comments': '<comments>',
             'inputformat': 'col'}  # 'row' or 'col'
# Shortcut dictionary to index benchmark functions by name
fitness_functions = {'sphere': free_function_ids[0], 'elipsoid': free_function_ids[1],
                     'rastrigin': free_function_ids[2], }



def sysPrint(string):
    """ Small function to take care of the 'overhead' of sys.stdout.write + flush """
    sys.stdout.write(string)
    sys.stdout.flush()


def mutateBitstring(individual):
    """ Extremely simple 1/n bit-flip mutation """
    bitstring = individual.dna
    n = len(bitstring)
    p = 1/n
    for i in range(n):
        if np.random.random() < p:
            bitstring[i] = 1-bitstring[i]


def mutateIntList(individual, num_options):
    """ extremely simple 1/n random integer mutation """

    Mut.adaptStepSize(individual)
    p = individual.baseStepSize + individual.stepSizeOffset

    int_list = individual.dna
    for i in range(individual.n):
        if np.random.random() < p:
            # -1 as random_integers is [1, val], -1 to simulate leaving out the current value
            new_int = np.random.random_integers(num_options[i]-1)-1
            if int_list[i] == new_int:
                new_int = num_options[i] - 1  # If we randomly selected the same value, pick the value we left out

            int_list[i] = new_int


def GA(n=10, budget=250, fitness_function='sphere'):
    """ Defines a Genetic Algorithm (GA) that evolves an Evolution Strategy (ES) for a given fitness function """

    # Where to store genotype-fitness information
    storage_file = open('GA_results_{}_{}.tdat'.format(n, fitness_function), 'w')

    # Fitness function to be passed on to the baseAlgorithm
    def fitnessFunction(bitstring):
        return evaluate_ES(bitstring=bitstring, fitness_function=fitness_function, storage_file=storage_file)

    # Assuming a dimensionality of 11 (8 boolean + 3 triples)
    GA_mu = 3
    GA_lambda = 12

    parameters = Parameters(n, budget, GA_mu, GA_lambda)
    # Initialize the first individual in the population
    population = [Individual(n)]
    population[0].dna = np.array([np.random.randint(len(x[1])) for x in options])
    population[0].fitness = fitnessFunction(population[0].dna)[0]

    while len(population) < GA_mu:
        population.append(copy(population[0]))

    # We use lambda functions here to 'hide' the additional passing of parameters that are algorithm specific
    functions = {
        'recombine': lambda pop: Rec.onePlusOne(pop),  # simply copy the only existing individual and return as a list
        'mutate': lambda ind: mutateIntList(ind, num_options),
        'select': lambda pop, new_pop, _: Sel.roulette(pop, new_pop, parameters),
        'mutateParameters': lambda t, _: parameters.oneFifthRule(t),
    }
    results = baseAlgorithm(population, fitnessFunction, budget, functions, parameters)
    storage_file.close()
    return results


def evaluate_ES(bitstring, fitness_function='sphere', opts=None, n=10, budget=None, storage_file=None):
    """ Single function to run all desired combinations of algorithms * fitness functions """

    # Set parameters
    if budget is None:
        budget = 10**2 * n
    num_runs = 15

    # Setup the bbob logger
    bbob_opts['algid'] = bitstring
    f = fgeneric.LoggingFunction(datapath, **bbob_opts)

    if opts:
        print(getBitString(opts))
    else:
        print(bitstring, end=' ')
        opts = getOpts(bitstring)

    # define local function of the algorithm to be used, fixing certain parameters
    def algorithm(n, evalfun, budget):
        return customizedES(n, evalfun, budget, opts=opts)

    '''
    # Actually running the algorithm is encapsulated in a try-except for now... math errors
    try:
        # Run the actual ES for <num_runs> times
        _, fitnesses = runAlgorithm(fitness_function, algorithm, n, num_runs, f, budget, opts)

        # From all different runs, retrieve the median fitness to be used as fitness for this ES
        min_fitnesses = np.min(fitnesses, axis=0)
        if storage_file:
            storage_file.write('{}:\t{}\n'.format(bitstring, min_fitnesses))
        median = np.median(min_fitnesses)
        print("\t\t{}".format(median))

        # mean_best_fitness = np.mean(min_fitnesses)
        # print(" {}  \t({})".format(mean_best_fitness, median))
    # '''

    _, fitnesses = runAlgorithm(fitness_function, algorithm, n, num_runs, f, budget, opts)

    # From all different runs, retrieve the median fitness to be used as fitness for this ES
    min_fitnesses = np.min(fitnesses, axis=0)
    if storage_file:
        storage_file.write('{}:\t{}\n'.format(bitstring, min_fitnesses))
    median = np.median(min_fitnesses)
    print("\t\t{}".format(median))


    '''
    except Exception as e:
        # Give this ES fitness INF in case of runtime errors
        print(" np.inf: {}".format(e))
        # mean_best_fitness = np.inf
        median = np.inf
    # '''
    return [median]


def fetchResults(fun_id, instance, n, budget, opts):
    """ Small overhead-function to enable multi-processing """
    f = fgeneric.LoggingFunction(datapath, **bbob_opts)
    f_target = f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=instance)).ftarget
    # Run the ES defined by opts once with the given budget
    results = customizedES(n, f.evalfun, budget, opts=opts)
    return f_target, results


def runAlgorithm(fit_name, algorithm, n, num_runs, f, budget, opts):

    fun_id = fitness_functions[fit_name]

    # Perform the actual run of the algorithm
    # '''  # Single-core version
    results = []
    targets = []
    for j in range(num_runs):
        # sysPrint('    Run: {}\r'.format(j))  # I want the actual carriage return here! No output clutter
        f_target = f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=j)).ftarget
        targets.append(f_target)
        results.append(algorithm(n, f.evalfun, budget))

    '''  # Multi-core version ## TODO: Fix using dill/pathos/something else
    from multiprocessing import Pool
    p = Pool(4)
    function = lambda x: fetchResults(fun_id, x, n, budget, opts)
    run_data = p.map(function, range(num_runs))
    targets, results = zip(*run_data)
    #'''

    # Preprocess/unpack results
    _, sigmas, fitnesses, best_individual = (list(x) for x in zip(*results))
    sigmas = np.array(sigmas).T
    # Subtract the target fitness value from all returned fitnesses to only get the absolute distance
    fitnesses = np.subtract(np.array(fitnesses).T, np.array(targets)[np.newaxis,:])

    return sigmas, fitnesses


def run():
    pass
    '''
    # Test all individual options
    n = len(options)
    evaluate_ES([0]*n)
    for i in range(n):
        for j in range(1, num_options[i]):
            dna = [0]*n
            dna[i] = j
            evaluate_ES(dna)

    print("\n\n")
    # '''

    '''
    # Known problems
    print("Combinations known to cause problems:")
    print("None! Good job :D")

    print("\n\n")
    # '''

    '''
    print("Mirrored vs Mirrored-pairwise")
    evaluate_ES(None, opts={'mirrored': True})
    evaluate_ES(None, opts={'mirrored': True, 'selection': 'pairwise'})
    # '''


    # '''
    # Exhaustive/brute-force search over *all* possible combinations
    # NB: THIS ASSUMES OPTIONS ARE SORTED ASCENDING BY NUMBER OF VALUES
    print("Brute-force exhaustive search of *all* available ES-combinations.")
    print("Number of possible ES-combinations currently available: {}".format(np.product(num_options)))
    from collections import Counter
    from itertools import product
    from datetime import datetime, timedelta

    best_ES = None
    best_result = np.inf

    products = []
    # count how often there is a choice of x options
    counts = Counter(num_options)
    for num, count in sorted(counts.items(), key=lambda x: x[0]):
        products.append(product(range(num), repeat=count))

    storage_file = open('bruteforce_10_sphere.tdat', 'w')
    x = datetime.now()
    for combo in product(*products):
        opts = list(sum(combo, ()))
        result = evaluate_ES(opts, storage_file=storage_file)[0]

        if result < best_result:
            best_result = result
            best_ES = opts

    y = datetime.now()


    print("Best ES found:       {}\n"
          "With median fitness: {}\n".format(best_ES, best_result))
    z = y - x
    days = z.days
    hours = z.seconds//3600
    minutes = (z.seconds % 3600) // 60
    seconds = (z.seconds % 60)

    print("Time at start:       {}\n"
          "Time at end:         {}\n"
          "Elapsed time:        {} days, {} hours, {} minutes, {} seconds".format(x, y, days, hours, minutes, seconds))
    # '''

    # '''
    pop, sigmas, fitness, best = GA()
    print()
    print("Best Individual:     {}\n"
          "        Fitness:     {}\n"
          "Fitnesses over time: {}".format(best.dna, best.fitness, fitness))
    # '''

if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)
    # np.random.seed(42)
    run()
