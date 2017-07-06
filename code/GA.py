#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import sys
from copy import copy
from datetime import datetime
from functools import partial
from itertools import product
from multiprocessing import Pool
from numpy import floor, log


from bbob import bbobbenchmarks, fgeneric
from code import getBitString, getOpts, getPrintName, getVals, options, num_options, num_threads, Config
from code import allow_parallel, MPI_available, MPI
from code.Algorithms import GA, MIES, customizedES
from code.Utils import ESFitness
from code.local import datapath, non_bbob_datapath

# BBOB parameters: Sets of noise-free and noisy benchmarks
free_function_ids = bbobbenchmarks.nfreeIDs
noisy_function_ids = bbobbenchmarks.noisyIDs

# Options to be stored in the log file(s)
bbob_opts = {'algid': None,
             'comments': '<comments>',
             'inputformat': 'col'}  # 'row' or 'col'


'''-----------------------------------------------------------------------------
#                            Small Utility Functions                           #
-----------------------------------------------------------------------------'''


def _sysPrint(string):
    """ Small function to take care of the 'overhead' of sys.stdout.write + flush """
    sys.stdout.write(string)
    sys.stdout.flush()


def _displayDuration(start, end):
    duration = end - start
    days = duration.days
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    seconds = (duration.seconds % 60)

    print("Time at start:       {}\n"
          "Time at end:         {}\n"
          "Elapsed time:        {} days, {} hours, {} minutes, {} seconds".format(start, end, days, hours, minutes, seconds))

    return duration


def _writeResultToFile(candidate, result, storage_file):
    if storage_file:
        with open(storage_file, 'a') as f:
            f.write(str("{}\t{}\n".format(candidate, repr(result))))
    print('\t', result)


def _trimFitnessHistoryByLength(fitnesses):
    fit_lengths = set([len(x) for x in fitnesses])
    if len(fit_lengths) > 1:
        min_length = min(fit_lengths)
        fitnesses = [x[:min_length] for x in fitnesses]

    return fitnesses


def _ensureListOfLists(iterable):
    try:
        if len(iterable) > 0:
            try:
                if len(iterable[0]) > 0:
                    return iterable
            except TypeError:
                return [iterable]
    except TypeError:
        return [[iterable]]


def _displayRepresentation(representation):
    disc_part = representation[:len(options)]
    lambda_ = representation[len(options)]
    mu = representation[len(options)+1]
    float_part = representation[len(options)+2:]

    print("{}({:.3f}, {}) with {}".format([int(x) for x in disc_part], mu, lambda_, float_part))


'''-----------------------------------------------------------------------------
#                          Old ES-Evaluation Functions                         #
-----------------------------------------------------------------------------'''


def ALT_evaluate_ES(bitstrings, fid, ndim, budget=None, storage_file=None, opts=None):
    """
        Single function to run all desired combinations of algorithms * fitness functions - MPI4PY VERSION

        :param bitstrings:      The genotype to be translated into customizedES-ready options. Must manually be set to
                                None if options are given as opts
        :param fid:             The BBOB function ID to use in the evaluation
        :param ndim:            The dimensionality to test the BBOB function with
        :param budget:          The allowed number of BBOB function evaluations
        :param storage_file:    Filename to use when storing fitness information
        :param opts:            Dictionary of options for customizedES. If omitted, the bitstring will be translated
                                into this options automatically
        :returns:               A list containing one instance of ESFitness representing the fitness of the defined ES
    """

    # Set parameters
    if budget is None:
        budget = Config.ES_budget_factor * ndim
    num_runs = Config.ES_num_runs
    num_ints = len(num_options) + 1
    fitness_results = []
    comms = []

    for bitstring in bitstrings:
        # Setup the bbob logger
        bbob_opts['algid'] = bitstring  # Save the bitstring of the ES we are currently evaluating

        print(bitstring)
        opts = getOpts(bitstring)
        values = getVals(bitstring[num_ints + 1:])

        function = partial(_fetchResults, fid, ndim=ndim, budget=budget, opts=opts, values=values)
        arguments = range(num_runs)

        comm = MPI.COMM_SELF.Spawn(sys.executable, args=['MPI_slave.py'], maxprocs=num_runs)  # Init
        comm.bcast(function, root=MPI.ROOT)  # Equal for all processes
        comm.scatter(arguments, root=MPI.ROOT)  # Different for each process
        comms.append(comm)

    for comm in comms:
        comm.Barrier()  # Wait for everything to finish...

    for i, comm in enumerate(comms):
        run_data = None  # Declaration of the variable to use for catching the data
        run_data = comm.gather(run_data, root=MPI.ROOT)  # And gather everything up
        comm.Disconnect()

        targets, results = zip(*run_data)

        # Preprocess/unpack results
        _, _, fitnesses, _ = (list(x) for x in zip(*results))
        fitnesses = _trimFitnessHistoryByLength(fitnesses)

        # Subtract the target fitness value from all returned fitnesses to only get the absolute distance
        fitnesses = np.subtract(np.array(fitnesses), np.array(targets).T[:, np.newaxis])
        fitness = ESFitness(fitnesses)
        fitness_results.append(fitness)

        if not isinstance(bitstrings[i], list):
            bitstrings[i] = bitstrings[i].tolist()
        _writeResultToFile(bitstrings[i], fitness, storage_file)

    return fitness_results


def evaluate_ES(es_genotype, fid, ndim, budget=None, storage_file=None, opts=None, values=None):
    """
        Single function to run all desired combinations of algorithms * fitness functions

        :param es_genotype:     The genotype to be translated into customizedES-ready options. Must manually be set to
                                None if options are given as opts
        :param fid:             The BBOB function ID to use in the evaluation
        :param ndim:            The dimensionality to test the BBOB function with
        :param budget:          The allowed number of BBOB function evaluations
        :param storage_file:    Filename to use when storing fitness information
        :param opts:            Dictionary of options for customizedES. If omitted, the bitstring will be translated
                                into this options automatically
        :param values:          Dictionary of initial values of parameters for the ES to be evaluated
        :returns:               A list containing one instance of ESFitness representing the fitness of the defined ES
    """

    # Set parameters
    if budget is None:
        budget = Config.ES_budget_factor * ndim
    num_runs = Config.ES_num_runs
    num_ints = len(num_options) + 1

    # Setup the bbob logger
    bbob_opts['algid'] = es_genotype  # Save the bitstring of the ES we are currently evaluating
    f = fgeneric.LoggingFunction(datapath, **bbob_opts)

    # If a dict of options is given, use that. Otherwise, translate the genotype to customizedES-ready options
    if opts:
        print(getBitString(opts), end=' ')
        lambda_ = opts['lambda_'] if 'lambda_' in opts else None
        mu = opts['mu'] if 'mu' in opts else None
    else:
        print(es_genotype[:num_ints + 1], end='')
        opts = getOpts(es_genotype[:num_ints - 1])
        lambda_ = es_genotype[num_ints - 1]
        mu = es_genotype[num_ints]
        values = getVals(es_genotype[num_ints + 1:])

    # define local function of the algorithm to be used, fixing certain parameters
    algorithm = partial(customizedES, lambda_=lambda_, mu=mu, opts=opts, values=values)

    # Run the actual ES for <num_runs> times
    fitnesses = runAlgorithm(fid, algorithm, ndim, num_runs, f, budget, opts, parallel=Config.ES_evaluate_parallel)
    fitness = ESFitness(fitnesses)

    _writeResultToFile(es_genotype, fitness, storage_file)

    return [fitness]


def runAlgorithm(fid, algorithm, ndim, num_runs, f, budget, opts, values=None, parallel=False):
    """
        Run the ES specified by ``opts`` and ``values`` on BBOB function ``fid`` in ``ndim``
        dimensionalities with ``budget``. Repeat ``num_runs`` times.

        ``algorithm`` and ``f`` are basically the same information, but are only used if this
        function is executed with ``parallel=False``

        :param fid:         BBOB function ID
        :param algorithm:   :func:`~code.Algorithms.customizedES` instance that is prepared using ``partial()``
        :param ndim:        Dimensionality to run the ES in
        :param num_runs:    Number of times to run the ES (in parallel) for calculating ERT/FCE
        :param f:           BBOB fitness function
        :param budget:      Evaluation budget for the ES
        :param opts:        Dictionary containing keyword options defining the ES-structure to run
        :param values:      Dictionary containing keyword-value pairs for the tunable parameters
        :param parallel:    Boolean, should the ES be run in parallel?
        :return:            Tuple(list of sigma-values over time, list of fitness-values over time)
    """

    function = partial(_fetchResults, fid, ndim=ndim, budget=budget, opts=opts, values=values)

    # Perform the actual run of the algorithm
    if parallel and Config.use_MPI:
        arguments = range(num_runs)
        run_data = None

        # mpi4py
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=['MPI_slave.py'], maxprocs=num_runs)  # Init
        comm.bcast(function, root=MPI.ROOT)  # Equal for all processes
        comm.scatter(arguments, root=MPI.ROOT)  # Different for each process
        comm.Barrier()  # Wait for everything to finish...
        run_data = comm.gather(run_data, root=MPI.ROOT)  # And gather everything up
        comm.Disconnect()

        targets, results = zip(*run_data)
    elif parallel and allow_parallel:  # Multi-core version
        p = Pool(min(num_threads, num_runs))
        run_data = p.map(function, range(num_runs))
        targets, results = zip(*run_data)
    else:  # Single-core version
        results = []
        targets = []
        for j in range(num_runs):
            f_target = f.setfun(*bbobbenchmarks.instantiate(fid, iinstance=j)).ftarget
            targets.append(f_target)
            results.append(algorithm(ndim, f.evalfun, budget))

    # Preprocess/unpack results
    _, _, fitnesses, _ = (list(x) for x in zip(*results))
    fitnesses = _trimFitnessHistoryByLength(fitnesses)

    # Subtract the target fitness value from all returned fitnesses to only get the absolute distance
    fitnesses = np.subtract(np.array(fitnesses), np.array(targets).T[:, np.newaxis])

    return fitnesses


def _fetchResults(fid, instance, ndim, budget, opts, values=None):
    """ Small overhead-function to enable multi-processing """
    f = fgeneric.LoggingFunction(datapath, **bbob_opts)
    f_target = f.setfun(*bbobbenchmarks.instantiate(fid, iinstance=instance)).ftarget
    # Run the ES defined by opts once with the given budget
    results = customizedES(ndim, f.evalfun, budget, opts=opts, values=values)
    return f_target, results


'''-----------------------------------------------------------------------------
#                          New ES-Evaluation Functions                         #
-----------------------------------------------------------------------------'''


def evaluateCustomizedESs(representations, iids, ndim, fid, budget=None, storage_file=None):
    """
        Function to evaluate customizedES instances using the BBOB framework. Can be passed one or more representations
        at once, will run them in parallel as much as possible if instructed to do so in Config.

        :param representations: The genotype to be translated into customizedES-ready options. Must manually be set to
                                None if options are given as opts
        :param iids:            The BBOB instance ID's to run the representation on (for statistical significance)
        :param ndim:            The dimensionality to test the BBOB function with
        :param fid:             The BBOB function ID to use in the evaluation
        :param budget:          The allowed number of BBOB function evaluations
        :param storage_file:    Filename to use when storing fitness information
        :returns:               A list containing one instance of ESFitness representing the fitness of the defined ES
    """

    # TODO: Expand this function to include number of repetitions per (ndim, fid, iid) combination?

    representations = _ensureListOfLists(representations)

    budget = Config.ES_budget_factor * ndim if budget is None else budget
    runFunction = partial(runCustomizedES, ndim=ndim, fid=fid, budget=budget)
    for rep in representations:
        _displayRepresentation(rep)
    arguments = product(representations, iids)

    if MPI_available and Config.use_MPI and Config.GA_evaluate_parallel:
        print("MPI run")
        run_data = runMPI(runFunction, list(arguments))
    elif allow_parallel and Config.GA_evaluate_parallel:
        print("Pool run")
        run_data = runPool(runFunction, list(arguments))
    else:
        print("Single-Threaded run")
        run_data = runSingleThreaded(runFunction, list(arguments))

    targets, results = zip(*run_data)
    fitness_results = []

    for i, rep in enumerate(representations):

        # Preprocess/unpack results
        _, _, fitnesses, _ = (list(x) for x in zip(*results[i*len(iids):(i+1)*len(iids)]))
        fitnesses = _trimFitnessHistoryByLength(fitnesses)

        # Subtract the target fitness value from all returned fitnesses to only get the absolute distance
        fitnesses = np.subtract(np.array(fitnesses), np.array(targets[i*len(iids):(i+1)*len(iids)]).T[:, np.newaxis])
        fitness = ESFitness(fitnesses)
        fitness_results.append(fitness)

        if not isinstance(rep, list):
            rep = rep.tolist()
        _writeResultToFile(rep, fitness, storage_file)

    return fitness_results


def runCustomizedES(representation, iid, ndim, fid, budget):
    """
        Runs a customized ES on a particular instance of a BBOB function in some dimensionality with given budget.
        This function takes care of the BBOB setup and the translation of the representation to input arguments for
        the customizedES.

        :param representation:  Representation of a customized ES (structure and parameter initialization)
        :param iid:             Instance ID for the BBOB function
        :param ndim:            Dimensionality to run the ES in
        :param fid:             BBOB function ID
        :param budget:          Evaluation budget for the ES
        :return:                Tuple(target optimum value of the evaluated function, list of fitness-values over time)
    """
    # Setup BBOB function + logging
    bbob_opts['algid'] = representation
    f = fgeneric.LoggingFunction(datapath, **bbob_opts)
    f_target = f.setfun(*bbobbenchmarks.instantiate(fid, iinstance=iid)).ftarget

    # Interpret the representation into parameters for the ES
    opts = getOpts(representation[:len(options)])
    lambda_ = representation[len(options)]
    mu = representation[len(options)+1]
    values = getVals(representation[len(options)+2:])

    # Run the ES defined by opts once with the given budget
    results = customizedES(ndim, f.evalfun, budget, lambda_=lambda_, mu=mu, opts=opts, values=values)
    return f_target, results


'''-----------------------------------------------------------------------------
#                       Parallelization-style Functions                        #
-----------------------------------------------------------------------------'''


def runMPI(runFunction, arguments):
    """
        Small overhead-function to handle multi-processing using MPI

        :param runFunction: The (``partial``) function to run in parallel, accepting ``arguments``
        :param arguments:   The arguments to passed distributedly to ``runFunction``
        :return:            List of any results produced by ``runFunction``
    """
    results = None  # Required pre-initialization of the variable that will receive the data from comm.gather()

    comm = MPI.COMM_SELF.Spawn(sys.executable, args=['MPI_slave.py'],
                               maxprocs=(min(Config.MPI_num_total_threads, len(arguments))))  # Initialize
    comm.bcast(runFunction, root=MPI.ROOT)          # Equal for all processes
    comm.scatter(arguments, root=MPI.ROOT)          # Different for each process
    comm.Barrier()                                  # Wait for everything to finish...
    results = comm.gather(results, root=MPI.ROOT)   # And gather everything up
    comm.Disconnect()

    return results


def runPool(runFunction, arguments):
    """
        Small overhead-function to handle multi-processing using Python's built-in multiprocessing.Pool

        :param runFunction: The (``partial``) function to run in parallel, accepting ``arguments``
        :param arguments:   The arguments to passed distributedly to ``runFunction``
        :return:            List of any results produced by ``runFunction``
    """
    p = Pool(min(num_threads, len(arguments)))
    results = p.map(runFunction, arguments)
    return results


def runSingleThreaded(runFunction, arguments):
    """
        Small overhead-function to iteratively run a function with a pre-determined input arguments

        :param runFunction: The (``partial``) function to run, accepting ``arguments``
        :param arguments:   The arguments to passed to ``runFunction``, one run at a time
        :return:            List of any results produced by ``runFunction``
    """
    results = []
    for arg in arguments:
        results.append(runFunction(*arg))
    return results


'''-----------------------------------------------------------------------------
#                                Run Functions                                 #
-----------------------------------------------------------------------------'''


def _testEachOption():
    # Test all individual options
    n = len(options)
    dna = [0] * n
    # lambda_mu = [None, None]
    lambda_mu = [2, 0.01]
    dna.extend(lambda_mu)
    evaluate_ES(dna, fid=1, ndim=10, )
    for i in range(n):
        for j in range(1, num_options[i]):
            dna = [0] * n
            dna[i] = j
            dna.extend(lambda_mu)
            evaluate_ES(dna, fid=1, ndim=10, )

    print("\n\n")


def _problemCases():
    # Known problems
    print("Combinations known to cause problems:")

    evaluate_ES(None, fid=1, ndim=10, opts={'sequential': True})
    evaluate_ES(None, fid=1, ndim=10, opts={'tpa': True})
    evaluate_ES(None, fid=1, ndim=10, opts={'selection': 'pairwise'})
    evaluate_ES(None, fid=1, ndim=10, opts={'tpa': True, 'selection': 'pairwise'})
    # these are the actual failures
    evaluate_ES(None, fid=1, ndim=10, opts={'sequential': True, 'selection': 'pairwise'})
    evaluate_ES(None, fid=1, ndim=10, opts={'sequential': True, 'tpa': True, 'selection': 'pairwise'})

    evaluate_ES([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 113, 0.18770573922911427], fid=1, ndim=10)
    evaluate_ES([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 107, 0.37768142336353183], fid=1, ndim=10)
    evaluate_ES([0, 1, 1, 0, 1, 0, 1, 1, 0, 2, 2, None, None], fid=1, ndim=10)
    evaluate_ES([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 27, 0.9383818903266666], fid=1, ndim=10)

    genotype = [0, 0, 1, 1, 0, 0, 1, 0, 1, 2, 2, 3, 0.923162952008686]
    print(getPrintName(getOpts(genotype[:-2])))
    evaluate_ES(genotype, fid=1, ndim=10)


def _exampleRuns():
    print("Mirrored vs Mirrored-pairwise")
    evaluate_ES(None, fid=1, ndim=10, opts={'mirrored': True})
    evaluate_ES(None, fid=1, ndim=10, opts={'mirrored': True, 'selection': 'pairwise'})

    print("Regular vs Active")
    evaluate_ES(None, fid=1, ndim=10, opts={'active': False})
    evaluate_ES(None, fid=1, ndim=10, opts={'active': True})

    print("No restart vs local restart")
    evaluate_ES(None, fid=1, ndim=10, opts={'ipop': None})
    evaluate_ES(None, fid=1, ndim=10, opts={'ipop': True})
    evaluate_ES(None, fid=1, ndim=10, opts={'ipop': 'IPOP'})
    evaluate_ES(None, fid=1, ndim=10, opts={'ipop': 'BIPOP'})


def _bruteForce(ndim, fid, parallel=1, part=0):
    # Exhaustive/brute-force search over *all* possible combinations
    # NB: THIS ASSUMES OPTIONS ARE SORTED ASCENDING BY NUMBER OF VALUES
    num_combinations = np.product(num_options)
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
        if start_at >= np.product(num_options):
            return  # Done.

    if part == 1 and start_at >= num_combinations // 2:  # Been there, done that
        return
    elif part == 2 and start_at < num_combinations // 2:  # THIS SHOULD NOT HAPPEN!!!
        print("{}\nWeird Error!\nstart_at smaller than intended!\n{}".format('-' * 32, '-' * 32))
        return

    products = []
    # count how often there is a choice of x options
    counts = Counter(num_options)
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
        if parallel == 1:
            result = evaluate_ES(bitstrings[0], fid=fid, ndim=ndim, storage_file=storage_file)
        else:
            result = ALT_evaluate_ES(bitstrings, fid=fid, ndim=ndim, storage_file=storage_file)

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
    # fitnessFunction = partial(ALT_evaluate_ES, fid=fid, ndim=ndim, storage_file=storage_file)
    # fitnessFunction = partial(evaluate_ES, fid=fid, ndim=ndim, storage_file=storage_file)

    fitnessFunction = partial(evaluateCustomizedESs, fid=fid, ndim=ndim, iids=range(Config.ES_num_runs), storage_file=storage_file)

    budget = Config.GA_budget

    gen_sizes, sigmas, fitness, best = MIES(n=ndim, fitnessFunction=fitnessFunction, budget=budget)  # This line does all the work!
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
            print("Optimizing for function ID {} in {}-dimensional space:".format(fid, ndim))
            x = datetime.now()
            gen_sizes, sigmas, fitness, best = MIES(ndim=ndim, fid=fid)
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
