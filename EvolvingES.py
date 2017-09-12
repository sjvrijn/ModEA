#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import sys
from functools import partial
from itertools import product
from multiprocessing import Pool

from bbob import bbobbenchmarks, fgeneric
from code import Config
from code.Algorithms import customizedES
from code.Utils import getOpts, getVals, options, initializable_parameters, \
    chunkListByLength, guaranteeFolderExists, reprToString, ESFitness
from code.local import datapath

try:
    from mpi4py import MPI
    MPI_available = True
except:
    MPI = None
    MPI_available = False

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
    """
        Display a human-readable time duration.

        :param start:   Time at the start
        :param end:     Time at the end
        :return:        The duration ``end - start``
    """
    duration = end - start
    days = duration.days
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    seconds = (duration.seconds % 60)

    print("Time at start:       {}\n"
          "Time at end:         {}\n"
          "Elapsed time:        {} days, {} hours, {} minutes, {} seconds".format(start, end, days, hours, minutes, seconds))

    return duration


def _writeResultToFile(representation, result, storage_file):
    """
        Log a representation and the result of its evaluation to a file.

        :param representation:  The representation
        :param result:          The evaluation result to be stored
        :param storage_file:    The filename to store it in. If ``None``, nothing happens
    """
    if storage_file:
        with open(storage_file, 'a') as f:
            f.write(str("{}\t{}\n".format(representation, repr(result))))
    print('\t', result)


def _trimListOfListsByLength(lists):
    """
        Given a list of lists of varying sizes, trim them to make the overall shape rectangular:

        >>> _trimListOfListsByLength([
        ...     [1,   2,   3,   4,   5],
        ...     [10,  20,  30],
        ...     ['a', 'b', 'c', 'd']
        ... ])
        [[1, 2, 3], [10, 20, 30], ['a', 'b', 'c']]

        :param lists:   The list of lists to trim
        :return:        The same lists, but trimmed in length to match the length of the shortest list from ``lists``
    """
    fit_lengths = set([len(x) for x in lists])
    if len(fit_lengths) > 1:
        min_length = min(fit_lengths)
        lists = [x[:min_length] for x in lists]

    return lists


def _ensureListOfLists(iterable):
    """
        Given an iterable, make sure it is at least a 2D array (i.e. list of lists):

        >>> _ensureListOfLists([[1, 2], [3, 4], [5, 6]])
        [[1, 2], [3, 4], [5, 6]]
        >>> _ensureListOfLists([1, 2])
        [[1, 2]]
        >>> _ensureListOfLists(1)
        [[1]]

        :param iterable:    The iterable of which to make sure it is 2D
        :return:            A guaranteed 2D version of ``iterable``
    """
    try:
        if len(iterable) > 0:
            try:
                if len(iterable[0]) > 0:
                    return iterable
            except TypeError:
                return [iterable]
    except TypeError:
        return [[iterable]]


def displayRepresentation(representation):
    """
        Displays a representation of a customizedES instance in a more human-readable format:

        >>> displayRepresentation([0,0,0,0,0,0,0,0,0,0,0,
        ...     20,0.25,1,1,1,1,1,1,0.2,0.955,0.5,0,0.3,0.5,2])
        [0,0,0,0,0,0,0,0,0,0,0] (0.25, 20) with [1,1,1,1,1,1,0.2,0.955,0.5,0,0.3,0.5,2]

        :param representation:  Representation of a customizedES instance to display
    """
    disc_part = representation[:len(options)]
    lambda_ = representation[len(options)]
    mu = '{:.3f}'.format(representation[len(options)+1]) if representation[len(options)+1] is not None else 'None'
    float_part = representation[len(options)+2:]

    print("{}({}, {}) with {}".format([int(x) for x in disc_part], mu, lambda_, float_part))


def ensureFullLengthRepresentation(representation):
    """
        Given a (partial) representation, ensure that it is padded to become a full length customizedES representation,
        consisting of the required number of structure, population and parameter values.

        >>> ensureFullLengthRepresentation([])
        [0,0,0,0,0,0,0,0,0,0,0, None,None, None,None,None,None,None,None,None,None,None,None,None,None,None]

        :param representation:  List representation of a customizedES instance to check and pad if needed
        :return:                Guaranteed full-length version of the representation
    """
    default_rep = [0]*len(options) + [None, None] + [None]*len(initializable_parameters)
    if len(representation) < len(default_rep):
        representation.extend(default_rep[len(representation):])
    return representation


'''-----------------------------------------------------------------------------
#                             ES-Evaluation Functions                          #
-----------------------------------------------------------------------------'''


def evaluateCustomizedESs(representations, iids, ndim, fid, budget=None, num_reps=1, storage_file=None):
    """
        Function to evaluate customizedES instances using the BBOB framework. Can be passed one or more representations
        at once, will run them in parallel as much as possible if instructed to do so in Config.

        :param representations: The genotype to be translated into customizedES-ready options.
        :param iids:            The BBOB instance ID's to run the representation on (for statistical significance)
        :param ndim:            The dimensionality to test the BBOB function with
        :param fid:             The BBOB function ID to use in the evaluation
        :param budget:          The allowed number of BBOB function evaluations
        :param num_reps:        Number of times each (ndim, fid, iid) combination has to be repeated
        :param storage_file:    Filename to use when storing fitness information
        :returns:               A list containing one instance of ESFitness representing the fitness of the defined ES
    """

    representations = _ensureListOfLists(representations)
    for rep in representations:
        displayRepresentation(rep)

    budget = Config.ES_budget_factor * ndim if budget is None else budget
    runFunction = partial(runCustomizedES, ndim=ndim, fid=fid, budget=budget)

    num_multiplications = len(iids*num_reps)
    arguments = list(product(representations, iids, range(num_reps)))
    run_data = runParallelFunction(runFunction, arguments)

    targets, results = zip(*run_data)
    fitness_results = []

    for i, rep in enumerate(representations):

        # Preprocess/unpack results
        _, _, fitnesses, _ = (list(x) for x in zip(*results[i*num_multiplications:(i+1)*num_multiplications]))
        fitnesses = _trimListOfListsByLength(fitnesses)

        # Subtract the target fitness value from all returned fitnesses to only get the absolute distance
        fitnesses = np.subtract(np.array(fitnesses), np.array(targets[i*num_multiplications:(i+1)*num_multiplications]).T[:, np.newaxis])
        fitness = ESFitness(fitnesses)
        fitness_results.append(fitness)

        if not isinstance(rep, list):
            rep = rep.tolist()
        _writeResultToFile(rep, fitness, storage_file)

    return fitness_results


def runCustomizedES(representation, iid, rep, ndim, fid, budget):
    """
        Runs a customized ES on a particular instance of a BBOB function in some dimensionality with given budget.
        This function takes care of the BBOB setup and the translation of the representation to input arguments for
        the customizedES.

        :param representation:  Representation of a customized ES (structure and parameter initialization)
        :param iid:             Instance ID for the BBOB function
        :param rep:             Repetition number (for output storage purposes only)
        :param ndim:            Dimensionality to run the ES in
        :param fid:             BBOB function ID
        :param budget:          Evaluation budget for the ES
        :return:                Tuple(target optimum value of the evaluated function, list of fitness-values over time)
    """
    # Setup BBOB function + logging
    bbob_opts['algid'] = representation
    datapath_ext = '{ndim}d-f{fid}/{repr}/i{iid}-r{rep}/'.format(ndim=ndim, fid=fid, repr=reprToString(representation),
                                                                 iid=iid, rep=rep)
    guaranteeFolderExists(datapath + datapath_ext)

    f = fgeneric.LoggingFunction(datapath + datapath_ext, **bbob_opts)
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


def runParallelFunction(runFunction, arguments):
    """
        Return the output of runFunction for each set of arguments,
        making use of as much parallelization as possible on this system

        :param runFunction: The function that can be executed in parallel
        :param arguments:   List of tuples, where each tuple are the arguments
                            to pass to the function
        :return:
    """
    if MPI_available and Config.use_MPI and Config.GA_evaluate_parallel:
        return runMPI(runFunction, arguments)
    elif Config.allow_parallel and Config.GA_evaluate_parallel:
        return runPool(runFunction, arguments)
    else:
        return runSingleThreaded(runFunction, arguments)


def runMPI(runFunction, arguments):
    """
        Small overhead-function to handle multi-processing using MPI

        :param runFunction: The function to run in parallel, accepting ``arguments``
        :param arguments:   The arguments to passed distributedly to ``runFunction``
        :return:            List of any results produced by ``runFunction``
    """
    results = []
    num_parallel = Config.MPI_num_total_threads

    for args in chunkListByLength(arguments, num_parallel):
        res = None  # Required pre-initialization of the variable that will receive the data from comm.gather()

        comm = MPI.COMM_SELF.Spawn(sys.executable, args=['MPI_slave.py'], maxprocs=len(args))  # Initialize
        comm.bcast(runFunction, root=MPI.ROOT)    # Equal for all processes
        comm.scatter(args, root=MPI.ROOT)         # Different for each process
        comm.Barrier()                            # Wait for everything to finish...
        res = comm.gather(res, root=MPI.ROOT)     # And gather everything up
        comm.Disconnect()

        results.extend(res)

    return results


# Inline function definition to allow the passing of multiple arguments to 'runFunction' through 'Pool.map'
def func_star(a_b, func):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return func(*a_b)


def runPool(runFunction, arguments):
    """
        Small overhead-function to handle multi-processing using Python's built-in multiprocessing.Pool

        :param runFunction: The (``partial``) function to run in parallel, accepting ``arguments``
        :param arguments:   The arguments to passed distributedly to ``runFunction``
        :return:            List of any results produced by ``runFunction``
    """
    p = Pool(min(Config.num_threads, len(arguments)))

    local_func = partial(func_star, func=runFunction)
    results = p.map(local_func, arguments)
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
