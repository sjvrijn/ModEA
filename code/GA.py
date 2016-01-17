#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import sys
from copy import copy
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from mpi4py import MPI


import code.Mutation as Mut
import code.Selection as Sel
import code.Recombination as Rec
from bbob import bbobbenchmarks, fgeneric
from code import allow_parallel, getOpts, getBitString, options, num_options, num_threads, Config
from code.Algorithms import customizedES, baseAlgorithm
from code.Individual import Individual
from code.Parameters import Parameters


# BBOB parameters: Sets of noise-free and noisy benchmarks
free_function_ids = bbobbenchmarks.nfreeIDs
noisy_function_ids = bbobbenchmarks.noisyIDs
datapath = "test_results/"  # Where to store results
non_bbob_datapath = "ga_results/"  # Where to store the results I personally generate
# Options to be stored in the log file(s)
bbob_opts = {'algid': None,
             'comments': '<comments>',
             'inputformat': 'col'}  # 'row' or 'col'
# Shortcut dictionary to index benchmark functions by name
fitness_functions = {'sphere': free_function_ids[0], 'elipsoid': free_function_ids[1],
                     'rastrigin': free_function_ids[2], }


def cleanResults(fid):
    import shutil
    shutil.rmtree('{}data_f{}'.format(datapath, fid))


def sysPrint(string):
    """ Small function to take care of the 'overhead' of sys.stdout.write + flush """
    sys.stdout.write(string)
    sys.stdout.flush()


def GA(n, budget=None, fit_func_id=1):
    """ Defines a Genetic Algorithm (GA) that evolves an Evolution Strategy (ES) for a given fitness function """

    # Where to store genotype-fitness information
    storage_file = '{}GA_results_{}dim_f{}.tdat'.format(non_bbob_datapath, n, fit_func_id)

    # Fitness function to be passed on to the baseAlgorithm
    fitnessFunction = partial(ALT_evaluate_ES, fit_func_id=fit_func_id, storage_file=storage_file)

    # Assuming a dimensionality of 11 (8 boolean + 3 triples)
    GA_mu = Config.GA_mu
    GA_lambda = Config.GA_lambda
    if budget is None:
        budget = Config.GA_budget

    parameters = Parameters(n, budget, GA_mu, GA_lambda)
    # Initialize the first individual in the population
    population = [Individual(n)]
    population[0].dna = np.array([np.random.randint(len(x[1])) for x in options])

    while len(population) < GA_mu:
        population.append(copy(population[0]))

    # We use functions here to 'hide' the additional passing of parameters that are algorithm specific
    recombine = partial(Rec.random, param=parameters)
    mutate = partial(Mut.mutateIntList, num_options=num_options)
    best = partial(Sel.best, param=parameters)
    def select(pop, new_pop, _):
        return best(pop, new_pop)
    def mutateParameters(t):
        pass  # The only actual parameter mutation is the self-adaptive step-size of each individual

    functions = {
        'recombine': recombine,
        'mutate': mutate,
        'select': select,
        'mutateParameters': mutateParameters,
    }
    # TODO FIXME: parallel currently causes ValueError: I/O operation on closed file
    results = baseAlgorithm(population, fitnessFunction, budget, functions, parameters,
                            parallel=Config.GA_parallel, debug=Config.GA_debug)
    return results


def ALT_evaluate_ES(bitstrings, fit_func_id, n, budget=None, storage_file=None, opts=None):
    """ Single function to run all desired combinations of algorithms * fitness functions """

    # Set parameters
    if budget is None:
        budget = Config.ES_budget_factor * n
    num_runs = Config.ES_num_runs
    parallel = Config.ES_parallel
    medians = []
    comms = []

    for bitstring in bitstrings:
        # Setup the bbob logger
        bbob_opts['algid'] = bitstring  # Save the bitstring of the ES we are currently evaluating
        f = fgeneric.LoggingFunction(datapath, **bbob_opts)

        print(bitstring)
        opts = getOpts(bitstring)

        function = partial(fetchResults, fit_func_id, n=n, budget=budget, opts=opts)
        arguments = range(num_runs)
        run_data = None

        # mpi4py
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=['code/MPI_slave.py'], maxprocs=num_runs)  # Init
        comm.bcast(function, root=MPI.ROOT)     # Equal for all processes
        comm.scatter(arguments, root=MPI.ROOT)  # Different for each process
        comms.append(comm)

    for comm in comms:
        comm.Barrier()

    for i, comm in enumerate(comms):
        # Wait for everything to finish...
        run_data = comm.gather(run_data, root=MPI.ROOT)  # And gather everything up
        comm.Disconnect()

        targets, results = zip(*run_data)

        # Preprocess/unpack results
        _, sigmas, fitnesses, best_individual = (list(x) for x in zip(*results))
        fit_lengths = set([len(x) for x in fitnesses])
        if len(fit_lengths) > 1:
            min_length = min(fit_lengths)
            fitnesses = [x[:min_length] for x in fitnesses]

        # Subtract the target fitness value from all returned fitnesses to only get the absolute distance
        fitnesses = np.subtract(np.array(fitnesses).T, np.array(targets)[np.newaxis,:])
        # From all different runs, retrieve the median fitness to be used as fitness for this ES
        min_fitnesses = np.min(fitnesses, axis=0)
        if not isinstance(bitstrings[i], list):
            bitstrings[i] = bitstrings[i].tolist()

        if storage_file:
            with open(storage_file, 'a') as f:
                f.write("{}\t{}\n".format(bitstrings[i], min_fitnesses.tolist()))
        median = np.median(min_fitnesses)
        print("\t{}".format(median))
        medians.append(median)

    return medians


def evaluate_ES(bitstring, fit_func_id=1, opts=None, n=10, budget=None, storage_file=None):
    """ Single function to run all desired combinations of algorithms * fitness functions """

    # Set parameters
    if budget is None:
        budget = Config.ES_budget_factor * n
    num_runs = Config.ES_num_runs

    # Setup the bbob logger
    bbob_opts['algid'] = bitstring  # Save the bitstring of the ES we are currently evaluating
    f = fgeneric.LoggingFunction(datapath, **bbob_opts)

    if opts:
        print(getBitString(opts))
    else:
        print(bitstring, end=' ')
        opts = getOpts(bitstring)

    # define local function of the algorithm to be used, fixing certain parameters
    algorithm = partial(customizedES, opts=opts)


    '''
    # Actually running the algorithm is encapsulated in a try-except for now... math errors
    try:
        # Run the actual ES for <num_runs> times
        _, fitnesses = runAlgorithm(fitness_function, algorithm, n, num_runs, f, budget, opts)

        # From all different runs, retrieve the median fitness to be used as fitness for this ES
        min_fitnesses = np.min(fitnesses, axis=0)
        if storage_file:
            with open(storage_file, 'a') as f:
                f.write("{}\t{}\n".format(bitstring.tolist(), min_fitnesses.tolist()))
        median = np.median(min_fitnesses)
        print("\t\t{}".format(median))

        # mean_best_fitness = np.mean(min_fitnesses)
        # print(" {}  \t({})".format(mean_best_fitness, median))
    # '''

    _, fitnesses = runAlgorithm(fit_func_id, algorithm, n, num_runs, f, budget, opts, parallel=Config.ES_parallel)

    # From all different runs, retrieve the median fitness to be used as fitness for this ES
    min_fitnesses = np.min(fitnesses, axis=0)
    if storage_file:
        with open(storage_file, 'a') as f:
            f.write("{}\t{}\n".format(bitstring.tolist(), min_fitnesses.tolist()))
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


def runAlgorithm(fit_func_id, algorithm, n, num_runs, f, budget, opts, parallel=False):

    # Perform the actual run of the algorithm
    if parallel and Config.use_MPI:
        function = partial(fetchResults, fit_func_id, n=n, budget=budget, opts=opts)
        arguments = range(num_runs)
        run_data = None

        # mpi4py
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=['MPI_slave.py'], maxprocs=num_runs)  # Init
        comm.bcast(function, root=MPI.ROOT)     # Equal for all processes
        comm.scatter(arguments, root=MPI.ROOT)  # Different for each process
        comm.Barrier()                          # Wait for everything to finish...
        run_data = comm.gather(run_data, root=MPI.ROOT)  # And gather everything up

        targets, results = zip(*run_data)
    elif parallel and allow_parallel:  # Multi-core version
        num_workers = min(num_threads, num_runs)
        function = partial(fetchResults, fit_func_id, n=n, budget=budget, opts=opts)

        # multiprocessing
        p = Pool(num_workers)
        run_data = p.map(function, range(num_runs))

        targets, results = zip(*run_data)
    else:  # Single-core version
        results = []
        targets = []
        for j in range(num_runs):
            # sysPrint('    Run: {}\r'.format(j))  # I want the actual carriage return here! No output clutter
            f_target = f.setfun(*bbobbenchmarks.instantiate(fit_func_id, iinstance=j)).ftarget
            targets.append(f_target)
            results.append(algorithm(n, f.evalfun, budget))

    # Preprocess/unpack results
    _, sigmas, fitnesses, best_individual = (list(x) for x in zip(*results))
    sigmas = np.array(sigmas).T

    fit_lengths = set([len(x) for x in fitnesses])
    if len(fit_lengths) > 1:
        min_length = min(fit_lengths)
        fitnesses = [x[:min_length] for x in fitnesses]

    # Subtract the target fitness value from all returned fitnesses to only get the absolute distance
    fitnesses = np.subtract(np.array(fitnesses).T, np.array(targets)[np.newaxis,:])

    return sigmas, fitnesses



################################################################################
################################ Run Functions #################################
################################################################################
def testEachOption():
    # Test all individual options
    n = len(options)
    evaluate_ES([0]*n)
    for i in range(n):
        for j in range(1, num_options[i]):
            dna = [0]*n
            dna[i] = j
            evaluate_ES(dna)

    print("\n\n")


def problemCases():
    # Known problems
    print("Combinations known to cause problems:")

    evaluate_ES(None, opts={'sequential': True})
    evaluate_ES(None, opts={'two-point': True})
    evaluate_ES(None, opts={'selection': 'pairwise'})
    evaluate_ES(None, opts={'two-point': True, 'selection': 'pairwise'})
    # these are the actual failures
    evaluate_ES(None, opts={'sequential': True, 'selection': 'pairwise'})
    evaluate_ES(None, opts={'sequential': True, 'two-point': True, 'selection': 'pairwise'})
    # print("None! Good job :D")

    print("\n\n")


def exampleRuns():
    print("Mirrored vs Mirrored-pairwise")
    evaluate_ES(None, opts={'mirrored': True})
    evaluate_ES(None, opts={'mirrored': True, 'selection': 'pairwise'})


def bruteForce(ndim, fid, parallel=1):
    # Exhaustive/brute-force search over *all* possible combinations
    # NB: THIS ASSUMES OPTIONS ARE SORTED ASCENDING BY NUMBER OF VALUES
    print("F{} in {} dimensions:".format(fid, ndim))
    print("Brute-force exhaustive search of *all* available ES-combinations.")
    print("Number of possible ES-combinations currently available: {}".format(np.product(num_options)))
    from collections import Counter
    from itertools import product
    from datetime import datetime

    best_ES = None
    best_result = np.inf

    products = []
    # count how often there is a choice of x options
    counts = Counter(num_options)
    for num, count in sorted(counts.items(), key=lambda x: x[0]):
        products.append(product(range(num), repeat=count))

    storage_file = '{}bruteforce_{}_f{}.tdat'.format(non_bbob_datapath, ndim, fid)
    x = datetime.now()

    all_combos = []
    for combo in list(product(*products)):
        all_combos.append(list(sum(combo, ())))

    num_iters = len(all_combos) // parallel
    num_iters += 0 if len(all_combos) % parallel == 0 else 1

    for i in range(num_iters):
        bitstrings = all_combos[i*parallel:(i+1)*parallel]

        result = ALT_evaluate_ES(bitstrings, fit_func_id=fid, n=ndim, storage_file=storage_file)
        cleanResults(fid)

        for j, res in enumerate(result):
            if res < best_result:
                best_result = res
                best_ES = bitstrings[j]

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


def runGA(ndim=10, fid=1):

    x = datetime.now()
    gen_sizes, sigmas, fitness, best = GA(n=ndim, fit_func_id=fid)  # This line does all the work!
    y = datetime.now()
    print()
    print("Best Individual:     {}\n"
          "        Fitness:     {}\n"
          "Fitnesses over time: {}".format(best.dna, best.fitness, fitness))
    z = y - x
    days = z.days
    hours = z.seconds//3600
    minutes = (z.seconds % 3600) // 60
    seconds = (z.seconds % 60)
    print("Time at start:       {}\n"
          "Time at end:         {}\n"
          "Elapsed time:        {} days, {} hours, {} minutes, {} seconds".format(x, y, days, hours, minutes, seconds))

    np.savez("{}final_GA_results_{}dim_f{}".format(non_bbob_datapath, ndim, fid),
             sigma=sigmas, best_fitness=fitness, best_result=best.dna, generation_sizes=gen_sizes, time_spent=z)


def runExperiments():
    for dim in Config.experiment_dims:
        for func_id in Config.experiment_funcs:
            print("Optimizing for function ID {} in {}-dimensional space:".format(func_id, dim))
            x = datetime.now()
            gen_sizes, sigmas, fitness, best = GA(n=dim, fit_func_id=func_id)
            y = datetime.now()

            z = y - x
            np.savez("{}final_GA_results_{}dim_f{}".format(non_bbob_datapath, dim, func_id),
                     sigma=sigmas, best_fitness=fitness, best_result=best.dna, generation_sizes=gen_sizes, time_spent=z)


def run():
    # testEachOption()
    # problemCases()
    # exampleRuns()
    # bruteForce()
    # runGA()
    runExperiments()
    pass


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)
    # np.random.seed(42)

    if len(sys.argv) == 3:
        ndim = int(sys.argv[1])
        fid = int(sys.argv[2])
        runGA(ndim, fid)
    elif len(sys.argv) == 4:
        ndim = int(sys.argv[1])
        fid = int(sys.argv[2])
        parallel = int(sys.argv[3])
        bruteForce(ndim, fid, parallel)
    else:
        run()
