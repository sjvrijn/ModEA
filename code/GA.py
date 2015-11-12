#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import sys
from bbob import bbobbenchmarks, fgeneric
from code import getOpts, options
from code.Algorithms import customizedES, baseAlgorithm
from code.Individual import Individual
from code.Parameters import Parameters
import code.Selection as Sel
import code.Recombination as Rec


# BBOB parameters: Sets of noise-free and noisy benchmarks
free_function_ids = bbobbenchmarks.nfreeIDs
noisy_function_ids = bbobbenchmarks.noisyIDs
datapath = "test_results/"  # Where to store results
bbob_opts = {'algid': None,
             'comments': '<comments>',
             'inputformat': 'col'}  # 'row' or 'col'
fitness_functions = {'sphere': free_function_ids[0], 'elipsoid': free_function_ids[1],
                     'rastrigin': free_function_ids[2], }



def sysPrint(string):
    """ Small function to take care of the 'overhead' of sys.stdout.write + flush """
    sys.stdout.write(string)
    sys.stdout.flush()


def mutateBitstring(individual):
    """ extremely simple 1/n mutation """
    bitstring = individual.dna
    n = len(bitstring)
    p = 1/n
    for i in range(n):
        if np.random.random() < p:
            bitstring[i] = 1-bitstring[i]


def GA(n=10, budget=100, fitness_function='sphere'):
    """ Defines a Genetic Algorithm (GA) that evolves an Evolution Strategy (ES) for a given fitness function """

    # fitnessFunction = lambda x: evaluate_ES(x, fitness_function)
    def fitnessFunction(x):
        return evaluate_ES(x, fitness_function)
    parameters = Parameters(n, 1, 3, budget)
    population = [Individual(n)]
    population[0].dna = np.random.randint(2, size=len(options))
    population[0].fitness = fitnessFunction(population[0].dna)[0]

    # We use lambda functions here to 'hide' the additional passing of parameters that are algorithm specific
    functions = {
        'recombine': lambda pop: Rec.onePlusOne(pop),  # simply copy the only existing individual and return as a list
        'mutate': mutateBitstring,
        'select': lambda pop, new_pop, _: Sel.best(pop, new_pop, parameters),
        'mutateParameters': lambda t: parameters.oneFifthRule(t),
    }

    return baseAlgorithm(population, fitnessFunction, budget, functions, parameters)


def evaluate_ES(bitstring, fitness_function='sphere'):
    """ Single function to run all desired combinations of algorithms * fitness functions """

    # Set parameters
    n = 10
    budget = 500
    num_runs = 15

    bbob_opts['algid'] = bitstring
    f = fgeneric.LoggingFunction(datapath, **bbob_opts)

    print(bitstring, end=' ')
    opts = getOpts(bitstring)
    # algorithm = lambda n, evalfun, budget: customizedES(n, evalfun, budget, opts=opts)
    def algorithm(n, evalfun, budget):
        return customizedES(n, evalfun, budget, opts=opts)

    try:
        _, fitnesses = runAlgorithm(fitness_function, algorithm, n, num_runs, f, budget, opts)

        min_fitnesses = np.min(fitnesses, axis=0)
        median = np.median(min_fitnesses)
        print("\t\t{}".format(median))

        # mean_best_fitness = np.mean(min_fitnesses)
        # print(" {}  \t({})".format(mean_best_fitness, median))

    except Exception as e:
        print(" np.inf: {}".format(e))
        # mean_best_fitness = np.inf
        median = np.inf

    return [median]


def fetchResults(fun_id, instance, n, budget, opts):
    """ Small overhead-function to enable multi-processing """
    f = fgeneric.LoggingFunction(datapath, **bbob_opts)
    f_target = f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=instance)).ftarget
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
    fitnesses = np.subtract(np.array(fitnesses).T, np.array(targets)[np.newaxis,:])

    return sigmas, fitnesses


def run():

    # Test all individual options
    # print(evaluate_ES([0,0,0,0,0,0,0,0]))
    # print(evaluate_ES([1,0,0,0,0,0,0,0]))
    # print(evaluate_ES([0,1,0,0,0,0,0,0]))
    # print(evaluate_ES([0,0,1,0,0,0,0,0]))
    # print(evaluate_ES([0,0,0,1,0,0,0,0]))
    # print(evaluate_ES([0,0,0,0,1,0,1,0]))
    # print(evaluate_ES([0,0,0,0,0,1,0,0]))
    # print(evaluate_ES([0,0,0,0,0,0,1,0]))
    # print(evaluate_ES([0,0,0,0,0,0,0,1]))

    pop, sigmas, fitness, best = GA()
    print()
    print("Best Individual:     {}\n"
          "        Fitness:     {}\n"
          "Fitnesses over time: {}".format(best.dna, best.fitness, fitness))


if __name__ == '__main__':
    np.set_printoptions(linewidth=200)
    # np.random.seed(42)
    run()
