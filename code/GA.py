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

    fitnessFunction = lambda x: evaluate_ES(x, fitness_function)
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
    budget = 10
    num_runs = 5

    bbob_opts['algid'] = bitstring
    f = fgeneric.LoggingFunction(datapath, **bbob_opts)

    print(bitstring, end=' ')
    opts = getOpts(bitstring)
    algorithm = lambda n, evalfun, budget: customizedES(n, evalfun, budget, opts=opts)

    try:
        _, fitnesses = runAlgorithm(fitness_function, algorithm, n, num_runs, f, budget)

        min_fitnesses = np.min(fitnesses, axis=0)
        median = np.median(min_fitnesses)
        mean_best_fitness = np.mean(min_fitnesses)
        print(" {}  \t({})".format(mean_best_fitness, median))
    except Exception as e:
        print(" np.inf: {}".format(e))
        mean_best_fitness = np.inf
        median = np.inf

    # return [mean_best_fitness]
    return [median]

def runAlgorithm(fit_name, algorithm, n, num_runs, f, budget):

    fun_id = fitness_functions[fit_name]
    results = []
    targets = []

    # Perform the actual run of the algorithm
    for j in range(num_runs):
        # sysPrint('    Run: {}\r'.format(j))  # I want the actual carriage return here! No output clutter
        f_target = f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=j)).ftarget
        targets.append(f_target)
        results.append(algorithm(n, f.evalfun, budget))

    # Preprocess/unpack results
    _, sigmas, fitnesses, best_individual = (list(x) for x in zip(*results))
    sigmas = np.array(sigmas).T
    fitnesses = np.subtract(np.array(fitnesses).T, np.array(targets)[np.newaxis,:])

    return sigmas, fitnesses



if __name__ == '__main__':
    # np.random.seed(42)

    # print(evaluate_ES([0,0,0,0,0,0,0]))
    # print(evaluate_ES([1,0,0,0,0,0,0]))
    # print(evaluate_ES([0,1,0,0,0,0,0]))
    # print(evaluate_ES([0,0,1,0,0,0,0]))
    # print(evaluate_ES([0,0,0,1,0,0,0]))
    # print(evaluate_ES([0,0,0,0,1,0,1]))
    # print(evaluate_ES([0,0,0,0,0,1,0]))
    # print(evaluate_ES([0,0,0,0,0,0,1]))

    pop, sigmas, fitness, best = GA()
    print()
    print("Best Individual:     {}\n"
          "        Fitness:     {}\n"
          "Fitnesses over time: {}".format(best.dna, best.fitness, fitness))

