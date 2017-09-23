#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import sys
from datetime import datetime
from functools import partial
from copy import copy
from bbob import bbobbenchmarks
from code import Config
from code.Algorithms import _MIES
from EvolvingES import ensureFullLengthRepresentation, evaluateCustomizedESs, _displayDuration
from code.Individual import MixedIntIndividual
from code.Parameters import Parameters
from code.Utils import ESFitness, getOpts, options, num_options_per_module, \
    getBitString, getPrintName, create_bounds, guaranteeFolderExists
from code.local import non_bbob_datapath

# Sets of noise-free and noisy benchmarks
free_function_ids = bbobbenchmarks.nfreeIDs
noisy_function_ids = bbobbenchmarks.noisyIDs
guaranteeFolderExists(non_bbob_datapath)


opts = {'algid': None,
        'comments': '<comments>',
        'inputformat': 'col'}  # 'row' or 'col'

def sysPrint(string):
    """ Small function to take care of the 'overhead' of sys.stdout.write + flush """
    sys.stdout.write(string)
    sys.stdout.flush()


def _testEachOption():
    # Test all individual options
    n = len(options)
    fid = 1
    ndim = 10
    representation = [0] * n
    lambda_mu = [None, None]
    representation.extend(lambda_mu)
    ensureFullLengthRepresentation(representation)
    evaluateCustomizedESs(representation, fid=fid, ndim=ndim, iids=range(Config.ES_num_runs))
    for i in range(n):
        for j in range(1, num_options_per_module[i]):
            representation = [0] * n
            representation[i] = j
            representation.extend(lambda_mu)
            ensureFullLengthRepresentation(representation)
            evaluateCustomizedESs(representation, fid=fid, ndim=ndim, iids=range(Config.ES_num_runs))

    print("\n\n")


def _problemCases():
    fid = 1
    ndim = 10
    iids = range(Config.ES_num_runs)

    # Known problems
    print("Combinations known to cause problems:")

    rep = ensureFullLengthRepresentation(getBitString({'sequential': True}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation(getBitString({'tpa': True}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation(getBitString({'selection': 'pairwise'}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation(getBitString({'tpa': True, 'selection': 'pairwise'}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    # these are the actual failures
    rep = ensureFullLengthRepresentation(getBitString({'sequential': True, 'selection': 'pairwise'}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation(getBitString({'sequential': True, 'tpa': True, 'selection': 'pairwise'}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)

    rep = ensureFullLengthRepresentation([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 113, 0.18770573922911427])
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 107, 0.37768142336353183])
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation([0, 1, 1, 0, 1, 0, 1, 1, 0, 2, 2, None, None])
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 27, 0.9383818903266666])
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)

    rep = ensureFullLengthRepresentation([0, 0, 1, 1, 0, 0, 1, 0, 1, 2, 2, 3, 0.923162952008686])
    print(getPrintName(getOpts(rep[:-2])))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)


def _exampleRuns():
    fid = 1
    ndim = 10
    iids = range(Config.ES_num_runs)

    print("Mirrored vs Mirrored-pairwise")
    rep = ensureFullLengthRepresentation(getBitString({'mirrored': True}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation(getBitString({'mirrored': True, 'selection': 'pairwise'}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)

    print("Regular vs Active")
    rep = ensureFullLengthRepresentation(getBitString({'active': False}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation(getBitString({'active': True}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)

    print("No restart vs local restart")
    rep = ensureFullLengthRepresentation(getBitString({'ipop': None}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation(getBitString({'ipop': True}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation(getBitString({'ipop': 'IPOP'}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)
    rep = ensureFullLengthRepresentation(getBitString({'ipop': 'BIPOP'}))
    evaluateCustomizedESs(rep, iids=iids, fid=fid, ndim=ndim)


def _bruteForce(ndim, fid, parallel=1, part=0):
    # Exhaustive/brute-force search over *all* possible combinations
    # NB: THIS ASSUMES OPTIONS ARE SORTED ASCENDING BY NUMBER OF VALUES
    num_combinations = np.product(num_options_per_module)
    print("F{} in {} dimensions:".format(fid, ndim))
    print("Brute-force exhaustive search of *all* available ES-combinations.")
    print("Number of possible ES-combinations currently available: {}".format(num_combinations))
    from collections import Counter
    from itertools import product
    from datetime import datetime
    import cPickle
    import os

    best_ES = None
    best_result = ESFitness()

    progress_log = '{}_f{}.prog'.format(ndim, fid)
    progress_fname = non_bbob_datapath + progress_log
    if progress_log not in os.listdir(non_bbob_datapath):
        start_at = 0
    else:
        with open(progress_fname) as progress_file:
            start_at = cPickle.load(progress_file)
        if start_at >= np.product(num_options_per_module):
            return  # Done.

    if part == 1 and start_at >= num_combinations // 2:  # Been there, done that
        return
    elif part == 2 and start_at < num_combinations // 2:  # THIS SHOULD NOT HAPPEN!!!
        print("{}\nWeird Error!\nstart_at smaller than intended!\n{}".format('-' * 32, '-' * 32))
        return

    products = []
    # count how often there is a choice of x options
    counts = Counter(num_options_per_module)
    for num, count in sorted(counts.items(), key=lambda x: x[0]):
        products.append(product(range(num), repeat=count))

    if Config.write_output:
        storage_file = '{}bruteforce_{}_f{}.tdat'.format(non_bbob_datapath, ndim, fid)
    else:
        storage_file = None
    x = datetime.now()

    all_combos = []
    for combo in list(product(*products)):
        all_combos.append(list(sum(combo, ())))

    if part == 0:
        num_cases = len(all_combos)
    elif part == 1:
        num_cases = len(all_combos) // 2 - start_at
    elif part == 2:
        num_cases = len(all_combos) - start_at
    else:
        return  # invalid 'part' value

    num_iters = num_cases // parallel
    num_iters += 0 if num_cases % parallel == 0 else 1

    for i in range(num_iters):
        bitstrings = all_combos[(start_at + i * parallel):(start_at + (i + 1) * parallel)]
        bitstrings = [ensureFullLengthRepresentation(bitstring) for bitstring in bitstrings]
        result = evaluateCustomizedESs(bitstrings, fid=fid, ndim=ndim, num_reps=10,
                                       iids=range(Config.ES_num_runs), storage_file=storage_file)

        with open(progress_fname, 'w') as progress_file:
            cPickle.dump((start_at + (i + 1) * parallel), progress_file)

        for j, res in enumerate(result):
            if res < best_result:
                best_result = res
                best_ES = bitstrings[j]

    y = datetime.now()

    print("Best ES found:       {}\n"
          "With fitness: {}\n".format(best_ES, best_result))

    _displayDuration(x, y)


def _runGA(ndim=5, fid=1, run=1):
    x = datetime.now()

    # Where to store genotype-fitness information
    # storage_file = '{}GA_results_{}dim_f{}.tdat'.format(non_bbob_datapath, ndim, fid)
    storage_file = '{}MIES_results_{}dim_f{}run_{}.tdat'.format(non_bbob_datapath, ndim, fid, run)

    # Fitness function to be passed on to the baseAlgorithm
    fitnessFunction = partial(evaluateCustomizedESs, fid=fid, ndim=ndim,
                              iids=range(Config.ES_num_runs), storage_file=storage_file)

    parameters = Parameters(len(options) + 15, Config.GA_budget, mu=Config.GA_mu, lambda_=Config.GA_lambda)
    parameters.l_bound[len(options):] = np.array([  2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(15)
    parameters.u_bound[len(options):] = np.array([200, 1, 5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5]).reshape(15)

    # Initialize the first individual in the population
    discrete_part = [np.random.randint(len(x[1])) for x in options]
    lamb = int(4 + np.floor(3 * np.log(parameters.n)))
    int_part = [lamb]
    float_part = [
        parameters.mu,
        parameters.alpha_mu, parameters.c_sigma, parameters.damps, parameters.c_c, parameters.c_1,
        parameters.c_mu,
        0.2, 0.955,
        0.5, 0, 0.3, 0.5,
        2
    ]

    population = [
        MixedIntIndividual(len(discrete_part) + len(int_part) + len(float_part),
                           num_discrete=len(num_options_per_module),
                           num_ints=len(int_part))
    ]
    population[0].genotype = np.array(discrete_part + int_part + float_part)
    population[0].fitness = ESFitness()

    while len(population) < Config.GA_mu:
        population.append(copy(population[0]))

    u_bound, l_bound = create_bounds(float_part, 0.3)
    parameters.u_bound[len(options) + 1:] = np.array(u_bound)
    parameters.l_bound[len(options) + 1:] = np.array(l_bound)

    gen_sizes, sigmas, fitness, best = _MIES(n=ndim, fitnessFunction=fitnessFunction, budget=Config.GA_budget,
                                             mu=Config.GA_mu, lambda_=Config.GA_lambda, parameters=parameters,
                                             population=population)  # This line does all the work!
    y = datetime.now()
    print()
    print("Best Individual:     {}\n"
          "        Fitness:     {}\n"
          "Fitnesses over time: {}".format(best.genotype, best.fitness, fitness))

    z = _displayDuration(x, y)

    if Config.write_output:
        np.savez("{}final_GA_results_{}dim_f{}_run{}".format(non_bbob_datapath, ndim, fid, run),
                 sigma=sigmas, best_fitness=fitness, best_result=best.genotype,
                 generation_sizes=gen_sizes, time_spent=z)


def _runExperiments():
    for ndim in Config.experiment_dims:
        for fid in Config.experiment_funcs:

            # Initialize the first individual in the population
            discrete_part = [np.random.randint(len(x[1])) for x in options]
            lamb = int(4 + np.floor(3 * np.log(parameters.n)))
            int_part = [lamb]
            float_part = [
                parameters.mu,
                parameters.alpha_mu, parameters.c_sigma, parameters.damps, parameters.c_c, parameters.c_1,
                parameters.c_mu,
                0.2, 0.955,
                0.5, 0, 0.3, 0.5,
                2
            ]

            population = [
                MixedIntIndividual(len(discrete_part) + len(int_part) + len(float_part),
                                   num_discrete=len(num_options_per_module),
                                   num_ints=len(int_part))
            ]
            population[0].genotype = np.array(discrete_part + int_part + float_part)
            population[0].fitness = ESFitness()

            while len(population) < Config.GA_mu:
                population.append(copy(population[0]))

            parameters = Parameters(len(options) + 15, Config.GA_budget, mu=Config.GA_mu, lambda_=Config.GA_lambda)
            parameters.l_bound[len(options):] = np.array([  2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(15)
            parameters.u_bound[len(options):] = np.array([200, 1, 5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5]).reshape(15)
            u_bound, l_bound = create_bounds(float_part, 0.3)
            parameters.u_bound[len(options) + 1:] = np.array(u_bound)
            parameters.l_bound[len(options) + 1:] = np.array(l_bound)

            print("Optimizing for function ID {} in {}-dimensional space:".format(fid, ndim))
            x = datetime.now()
            gen_sizes, sigmas, fitness, best = _MIES(n=ndim, fitnessFunction=fid, budget=Config.GA_budget,
                                                     mu=Config.GA_mu, lambda_=Config.GA_lambda, parameters=parameters,
                                                     population=population)
            y = datetime.now()

            z = y - x
            np.savez("{}final_GA_results_{}dim_f{}".format(non_bbob_datapath, ndim, fid),
                     sigma=sigmas, best_fitness=fitness, best_result=best.genotype,
                     generation_sizes=gen_sizes, time_spent=z)


def runDefault():
    _runGA()
    # _testEachOption()
    # _problemCases()
    # _exampleRuns()
    # _bruteForce(ndim=10, fid=1)
    # _runExperiments()
    pass


def main():
    np.set_printoptions(linewidth=1000, precision=3)

    if len(sys.argv) == 3:
        ndim = int(sys.argv[1])
        fid = int(sys.argv[2])
        _runGA(ndim, fid)
    elif len(sys.argv) == 4:
        ndim = int(sys.argv[1])
        fid = int(sys.argv[2])
        run = int(sys.argv[3])
        _runGA(ndim, fid, run)
    elif len(sys.argv) == 5:
        ndim = int(sys.argv[1])
        fid = int(sys.argv[2])
        parallel = int(sys.argv[3])
        part = int(sys.argv[4])
        _bruteForce(ndim, fid, parallel, part)
    else:
        runDefault()


if __name__ == '__main__':
    main()
