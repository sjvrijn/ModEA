#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import sys
from copy import copy
from datetime import datetime
from functools import partial, total_ordering
from multiprocessing import Pool
from mpi4py import MPI


import code.Mutation as Mut
import code.Selection as Sel
import code.Recombination as Rec
from bbob import bbobbenchmarks, fgeneric
from code import allow_parallel, getBitString, getOpts, getPrintName, getVals, options, num_options, num_threads, Config
from code.Algorithms import customizedES, baseAlgorithm
from code.Individual import MixedIntIndividual
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


# TODO: Move this to a 'utils'-like file?
@total_ordering
class ESFitness(object):
    """
        Object to calculate and store the fitness measure for an ES and allow easy comparison.
        This measure consists of both the always available Fixed Cost Error (FCE)
        and the less available but more rigorous Expected Running Time (ERT).
    """
    def __init__(self, fitnesses=None, target=Config.default_target,         # Original values
                 min_fitnesses=None, min_indices=None, num_successful=None,  # Summary values
                 ERT=None, FCE=float('inf'), std_dev=None):                  # Human-readable values

        # If original fitness values are given, calculate everything from scratch
        if fitnesses is not None:
            min_fitnesses, min_indices, num_successful = self._preCalcFCEandERT(fitnesses, target)

        # If 'summary data' is available, calculate ERT, FCE and its std_dev using the summary data
        if min_fitnesses is not None and min_indices is not None and num_successful is not None:
            ERT, FCE, std_dev = self._calcFCEandERT(min_fitnesses, min_indices, num_successful)

        # The interesting values to display or use as comparison
        self.ERT = ERT          # Expected Running Time
        self.FCE = FCE          # Fixed Cost Error
        self.std_dev = std_dev  # Standard deviation of FCE
        # Summary/memory values to use for reproducability
        self.min_fitnesses = min_fitnesses
        self.min_indices = min_indices
        self.num_successful = num_successful
        self.target = target

    def __eq__(self, other):
        if self.ERT is not None and self.ERT == other.ERT:
            return True  # Both have a valid and equal ERT value
        elif self.ERT is None and other.ERT is None and self.FCE == other.FCE:
            return True  # Neither have a valid ERT value, but FCE is equal
        else:
            return False

    def __lt__(self, other):  # Assuming minimalization problems, so A < B means A is better than B
        if self.ERT is not None and other.ERT is None:
            return True  # If only one has an ERT, it is better by default
        elif self.ERT is not None and other.ERT is not None and self.ERT < other.ERT:
            return True  # If both have an ERT, we want the better one
        elif self.ERT is None and other.ERT is None and self.FCE < other.FCE:
            return True  # If neither have an ERT, we want the better FCE
        else:
            return False

    def __repr__(self):
        if self.min_fitnesses is not None:
            kwargs = "target={},min_fitnesses={},min_indices={},num_successful={}".format(
                self.target, self.min_fitnesses, self.min_indices, self.num_successful
            )
        else:
            kwargs = "target={},ERT={},FCE={},std_dev={}".format(self.target, self.ERT, self.FCE, self.std_dev)
        return "ESFitness({})".format(kwargs)

    def __unicode__(self):
        # TODO: pretty-print-ify
        return "ERT: {0:.6}  \tFCE: {1:.4}  \t(std: {2:.4})".format(self.ERT, self.FCE, self.std_dev)

    __str__ = __unicode__


    @staticmethod
    def _preCalcFCEandERT(fitnesses, target):
        """
            Calculates the FCE and ERT of a given set of function evaluation results and target value

            :param fitnesses:   Numpy array of size (num_runs, num_evals)
            :param target:      Target value to use for basing the ERT on. Default: 1e-8
            :return:            ESFitness object with FCE and ERT properly set
        """
        min_fitnesses = np.min(fitnesses, axis=1).tolist()

        num_runs, num_evals = fitnesses.shape
        below_target = fitnesses < target
        num_below_target = np.sum(below_target, axis=1)
        min_indices = []
        num_successful = 0
        for i in range(num_runs):
            if num_below_target[i] != 0:
                # Take the lowest index at which the target was reached.
                min_index = np.min(np.argwhere(below_target[i]))
                num_successful += 1
            else:
                # No evaluation reached the target in this run
                min_index = num_evals
            min_indices.append(min_index)

        return min_fitnesses, min_indices, num_successful


    @staticmethod
    def _calcFCEandERT(min_fitnesses, min_indices, num_successful):
        """
            Calculates the FCE and ERT of a given set of function evaluation results and target value

            :param min_fitnesses:   Numpy array
            :param min_indices:     List
            :param num_successful:  Integer
            :return:
        """

        ### FCE ###
        FCE = np.median(min_fitnesses)
        std_dev = np.std(min_fitnesses)

        ### ERT ###
        # If none of the runs reached the target, there is no (useful) ERT to be calculated
        ERT = np.sum(min_indices) / num_successful if num_successful != 0 else None

        return ERT, FCE, std_dev


def cleanResults(fid):
    import os
    import shutil
    shutil.rmtree('{}data_f{}'.format(datapath, fid))
    os.remove("{}bbobexp_f{}.info".format(datapath, fid))

def sysPrint(string):
    """ Small function to take care of the 'overhead' of sys.stdout.write + flush """
    sys.stdout.write(string)
    sys.stdout.flush()


def GA(ndim, fid, budget=None):
    """ Defines a Genetic Algorithm (GA) that evolves an Evolution Strategy (ES) for a given fitness function """

    # Where to store genotype-fitness information
    storage_file = '{}GA_results_{}dim_f{}.tdat'.format(non_bbob_datapath, ndim, fid)

    # Fitness function to be passed on to the baseAlgorithm
    # fitnessFunction = partial(ALT_evaluate_ES, fid=fid, ndim=ndim, storage_file=storage_file)
    fitnessFunction = partial(evaluate_ES, fid=fid, ndim=ndim, storage_file=storage_file)

    # Assuming a dimensionality of 11 (8 boolean + 3 triples)
    GA_mu = Config.GA_mu
    GA_lambda = Config.GA_lambda
    if budget is None:
        budget = Config.GA_budget

    parameters = Parameters(len(options) + 15, budget, mu=GA_mu, lambda_=GA_lambda)
    parameters.l_bound[len(options):] = np.array([2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(15,1)
    parameters.u_bound[len(options):] = np.array([200, 1, 5, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5]).reshape(15,1)
    # Initialize the first individual in the population
    population = [MixedIntIndividual(ndim, num_ints=len(num_options)+1)]
    int_part = [np.random.randint(len(x[1])) for x in options]
    int_part.append(None)
    # TODO FIXME: dumb, brute force, hardcoded defaults for testing purposes
    float_part = [None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    # float_part = [None, 2,    None, None, None, None, None, 0.2, 0.995, 0.5,  0,    0.3,  0.5,  2]

    population[0].genotype = np.array(int_part + float_part)
    population[0].fitness = ESFitness(FCE=np.inf)

    while len(population) < GA_mu:
        population.append(copy(population[0]))

    # We use functions here to 'hide' the additional passing of parameters that are algorithm specific
    recombine = Rec.random
    mutate = partial(Mut.mutateMixedInteger, options=options, num_options=num_options)
    best = Sel.bestGA
    def select(pop, new_pop, _, params):
        return best(pop, new_pop, params)
    def mutateParameters(t):
        pass  # The only actual parameter mutation is the self-adaptive step-size of each individual

    functions = {
        'recombine': recombine,
        'mutate': mutate,
        'select': select,
        'mutateParameters': mutateParameters,
    }
    # TODO FIXME: parallel currently causes ValueError: I/O operation on closed file
    _, results = baseAlgorithm(population, fitnessFunction, budget, functions, parameters,
                               parallel=Config.GA_parallel, debug=Config.GA_debug)
    return results


def ALT_evaluate_ES(bitstrings, fid, ndim, budget=None, storage_file=None, opts=None):
    """ Single function to run all desired combinations of algorithms * fitness functions """

    # Set parameters
    if budget is None:
        budget = Config.ES_budget_factor * ndim
    num_runs = Config.ES_num_runs
    parallel = Config.ES_parallel
    fitness_results = []
    comms = []

    for bitstring in bitstrings:
        # Setup the bbob logger
        bbob_opts['algid'] = bitstring  # Save the bitstring of the ES we are currently evaluating
        f = fgeneric.LoggingFunction(datapath, **bbob_opts)  # TODO: Why is this here when it is done in fetchResults?

        print(bitstring)
        opts = getOpts(bitstring)

        function = partial(fetchResults, fid, ndim=ndim, budget=budget, opts=opts)
        arguments = range(num_runs)

        # mpi4py
        comm = MPI.COMM_SELF.Spawn(sys.executable, args=['code/MPI_slave.py'], maxprocs=num_runs)  # Init
        comm.bcast(function, root=MPI.ROOT)     # Equal for all processes
        comm.scatter(arguments, root=MPI.ROOT)  # Different for each process
        comms.append(comm)

    for comm in comms:
        comm.Barrier()  # Wait for everything to finish...

    for i, comm in enumerate(comms):
        run_data = None                                  # Declaration of the variable to use for catching the data
        run_data = comm.gather(run_data, root=MPI.ROOT)  # And gather everything up
        # run_data = comm.gather(root=MPI.ROOT)  # And gather everything up
        comm.Disconnect()

        targets, results = zip(*run_data)

        # Preprocess/unpack results
        _, sigmas, fitnesses, best_individual = (list(x) for x in zip(*results))
        fit_lengths = set([len(x) for x in fitnesses])
        if len(fit_lengths) > 1:
            min_length = min(fit_lengths)
            fitnesses = [x[:min_length] for x in fitnesses]

        # TODO: replace by use of ESFitness objects
        # Subtract the target fitness value from all returned fitnesses to only get the absolute distance
        fitnesses = np.subtract(np.array(fitnesses), np.array(targets).T[:,np.newaxis])
        fitness = ESFitness(fitnesses)
        fitness_results.append(fitness)

        if not isinstance(bitstrings[i], list):
            bitstrings[i] = bitstrings[i].tolist()

        if storage_file:
            with open(storage_file, 'a') as f:
                f.write(str("{}\t{}\n".format(bitstrings[i], repr(fitness))))
        print('\t', fitness)

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
    num_ints = len(num_options)+1

    # Setup the bbob logger
    bbob_opts['algid'] = es_genotype  # Save the bitstring of the ES we are currently evaluating
    f = fgeneric.LoggingFunction(datapath, **bbob_opts)

    # If a dict of options is given, use that. Otherwise, translate the genotype to customizedES-ready options
    if opts:
        print(getBitString(opts), end=' ')
        lambda_ = opts['lambda_'] if 'lambda_' in opts else None
        mu = opts['mu'] if 'mu' in opts else None
    else:
        print(es_genotype[:num_ints+1], end=' ')
        opts = getOpts(es_genotype[:num_ints-1])
        lambda_ = es_genotype[num_ints-1]
        mu = es_genotype[num_ints]
        values = getVals(es_genotype[num_ints+1:])

    # define local function of the algorithm to be used, fixing certain parameters
    algorithm = partial(customizedES, lambda_=lambda_, mu=mu, opts=opts, values=values)

    # Run the actual ES for <num_runs> times
    _, fitnesses = runAlgorithm(fid, algorithm, ndim, num_runs, f, budget, opts, parallel=Config.ES_parallel)

    fitness = ESFitness(fitnesses)
    if storage_file:
        with open(storage_file, 'a') as f:
            f.write(str("{}\n".format(repr(fitness))))
    print('\t', fitness)
    return [fitness]

def fetchResults(fid, instance, ndim, budget, opts, values=None):
    """ Small overhead-function to enable multi-processing """
    f = fgeneric.LoggingFunction(datapath, **bbob_opts)
    f_target = f.setfun(*bbobbenchmarks.instantiate(fid, iinstance=instance)).ftarget
    # Run the ES defined by opts once with the given budget
    results = customizedES(ndim, f.evalfun, budget, opts=opts, values=values)
    return f_target, results


def runAlgorithm(fid, algorithm, ndim, num_runs, f, budget, opts, parallel=False):

    # Perform the actual run of the algorithm
    if parallel and Config.use_MPI:
        function = partial(fetchResults, fid, n=ndim, budget=budget, opts=opts)
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
        function = partial(fetchResults, fid, ndim=ndim, budget=budget, opts=opts)

        # multiprocessing
        p = Pool(num_workers)
        run_data = p.map(function, range(num_runs))

        targets, results = zip(*run_data)
    else:  # Single-core version
        results = []
        targets = []
        for j in range(num_runs):
            # sysPrint('    Run: {}\r'.format(j))  # I want the actual carriage return here! No output clutter
            f_target = f.setfun(*bbobbenchmarks.instantiate(fid, iinstance=j)).ftarget
            targets.append(f_target)
            results.append(algorithm(ndim, f.evalfun, budget))

    # Preprocess/unpack results
    _, sigmas, fitnesses, best_individual = (list(x) for x in zip(*results))
    sigmas = np.array(sigmas).T

    fit_lengths = set([len(x) for x in fitnesses])
    if len(fit_lengths) > 1:
        min_length = min(fit_lengths)
        fitnesses = [x[:min_length] for x in fitnesses]

    # Subtract the target fitness value from all returned fitnesses to only get the absolute distance
    fitnesses = np.subtract(np.array(fitnesses), np.array(targets).T[:,np.newaxis])

    return sigmas, fitnesses



'''-----------------------------------------------------------------------------
#                                Run Functions                                 #
-----------------------------------------------------------------------------'''
def testEachOption():
    # Test all individual options
    n = len(options)
    dna = [0]*n
    # lambda_mu = [None, None]
    lambda_mu = [2, 0.01]
    dna.extend(lambda_mu)
    evaluate_ES(dna, fid=1, ndim=10,)
    for i in range(n):
        for j in range(1, num_options[i]):
            dna = [0]*n
            dna[i] = j
            dna.extend(lambda_mu)
            evaluate_ES(dna, fid=1, ndim=10,)

    print("\n\n")


def problemCases():
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

    dna = [0, 0, 1, 1, 0, 0, 1, 0, 1, 2, 2, 3, 0.923162952008686]
    print(getPrintName(getOpts(dna[:-2])))
    evaluate_ES(dna, fid=1, ndim=10)

    print("None! Good job :D\n")


def exampleRuns():
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


def bruteForce(ndim, fid, parallel=1, part=0):
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

    progress_log = 'progress-f{}-{}dim.log'.format(fid, ndim)
    progress_fname = "{}{}".format(non_bbob_datapath, progress_log)
    if progress_log not in os.listdir(non_bbob_datapath):
        start_at = 0
    else:
        with open(progress_fname) as progress_file:
            start_at = cPickle.load(progress_file)
        if start_at == np.product(num_options):
            return  # Done.

    if part == 1 and start_at >= num_combinations // 2:  # Been there, done that
        return
    elif part == 2 and start_at < num_combinations // 2:  # THIS SHOULD NOT HAPPEN!!!
        print("{}\nWeird Error!\nstart_at smaller than intended!\n{}".format('-'*32, '-'*32))
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
        num_cases = len(all_combos)//2 - start_at
    elif part == 2:
        num_cases = len(all_combos) - start_at
    else:
        return  # invalid 'part' value

    num_iters = num_cases // parallel
    num_iters += 0 if num_cases % parallel == 0 else 1

    for i in range(num_iters):
        bitstrings = all_combos[(start_at + i*parallel):(start_at + (i+1)*parallel)]
        if parallel == 1:
            bitstrings[0].extend([2, 0.01])
            result = evaluate_ES(bitstrings[0], fid=fid, ndim=ndim, storage_file=storage_file)
        else:
            result = ALT_evaluate_ES(bitstrings, fid=fid, ndim=ndim, storage_file=storage_file)

        with open(progress_fname, 'w') as progress_file:
            cPickle.dump((start_at + (i+1)*parallel), progress_file)
        if Config.write_output:
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
    gen_sizes, sigmas, fitness, best = GA(ndim=ndim, fid=fid)  # This line does all the work!
    y = datetime.now()
    print()
    print("Best Individual:     {}\n"
          "        Fitness:     {}\n"
          "Fitnesses over time: {}".format(best.genotype, best.fitness, fitness))
    z = y - x
    days = z.days
    hours = z.seconds//3600
    minutes = (z.seconds % 3600) // 60
    seconds = (z.seconds % 60)
    print("Time at start:       {}\n"
          "Time at end:         {}\n"
          "Elapsed time:        {} days, {} hours, {} minutes, {} seconds".format(x, y, days, hours, minutes, seconds))

    if Config.write_output:
        np.savez("{}final_GA_results_{}dim_f{}".format(non_bbob_datapath, ndim, fid),
                 sigma=sigmas, best_fitness=fitness, best_result=best.genotype,
                 generation_sizes=gen_sizes, time_spent=z)


def runExperiments():
    for ndim in Config.experiment_dims:
        for fid in Config.experiment_funcs:
            print("Optimizing for function ID {} in {}-dimensional space:".format(fid, ndim))
            x = datetime.now()
            gen_sizes, sigmas, fitness, best = GA(ndim=ndim, fid=fid)
            y = datetime.now()

            z = y - x
            np.savez("{}final_GA_results_{}dim_f{}".format(non_bbob_datapath, ndim, fid),
                     sigma=sigmas, best_fitness=fitness, best_result=best.genotype,
                     generation_sizes=gen_sizes, time_spent=z)


def run():
    # testEachOption()
    # problemCases()
    # exampleRuns()
    bruteForce(ndim=10, fid=1)
    runGA()
    # runExperiments()
    pass


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)
    # np.seterr(all='raise')
    # np.random.seed(42)

    # HARDCODED: JUST GIVE US THE EXACT TARGET IN THESE CASES TODO: does not work???
    fgeneric.deltaftarget = 0
    fgeneric.write_output = Config.write_output

    if len(sys.argv) == 3:
        ndim = int(sys.argv[1])
        fid = int(sys.argv[2])
        runGA(ndim, fid)
    elif len(sys.argv) == 4:
        ndim = int(sys.argv[1])
        fid = int(sys.argv[2])
        parallel = int(sys.argv[3])
        bruteForce(ndim, fid, parallel)
    elif len(sys.argv) == 5:
        ndim = int(sys.argv[1])
        fid = int(sys.argv[2])
        parallel = int(sys.argv[3])
        part = int(sys.argv[4])
        bruteForce(ndim, fid, parallel, part)
    else:
        run()
