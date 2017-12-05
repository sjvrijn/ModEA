#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is written by F.Ye
# To do test

import urllib
import tarfile
import glob
#from pylab import *
#import bbob.bbob_pproc as bb
#import bbob.bbob_pproc.pprldistr as pprldistr


from code import Config
from functools import partial
from itertools import product
from bbob import bbobbenchmarks, fgeneric
from EvolvingES import runCustomizedES,evaluateCustomizedESs,_ensureListOfLists,_trimListOfListsByLength,runParallelFunction,ESFitness,\
    displayRepresentation,reorganiseBBOBOutput,reprToString,ensureFullLengthRepresentation
from code.local import datapath

def _evaluateESSpace(iid,ndim,fid,rep,budget):
    storage_file = 'Result'
    representation = [0,0,0,0,0,0,0,0,0,0,0,20,0.25,1,1,1,1,1,1,0.2,0.955,0.5,0,0.3,0.5,2]
    #fitness_result = EvEs.evaluateCustomizedESs(representation,iids=range(Config.ES_num_runs),ndim=Config.experiment_dims,fid=1,budget=Config.GA_budget,num_reps=100,storage_file=storage_file)
    #run_data = runCustomizedES(representation=representation,iid=iid,ndim=ndim, fid=fid, rep =rep,budget=budget)
    run_data = _runES(representation=representation,iid=iid,ndim=ndim, fid=fid, rep =rep,budget=budget)
    target = run_data[0]
    results = run_data[1]
    rtarget = [False] * rep
    rtarget = [rtarget] * len(results)

    space = 0.0
    for j in rep:
        count = 0
        for i in len(results):
            if j == 0:
                if results[i][j] <= target[i]:
                    rtarget[i][j] = True
                    count = count + 1
                else:
                    rtarget[i][j] = False
                if rtarget[i][j-1] == True:
                    rtarget[i][j] = True
                    count = count + 1
                else:
                    if results[i][j] <= target[i]:
                        rtarget[i][j] = True
                        count = count + 1
                    else:
                        rtarget[i][j] = False
        space = space + (count / len(results))

    return space

def _runES(representation, iid, rep, ndim, fid, budget,runtimes=10):
    count = 0
    target = []
    results = []
    while count < runtimes:
        run_data = runCustomizedES(representation=representation,iid=iid,ndim=ndim,fid=fid,rep=rep,budget=budget)
        t = run_data[0]
        r = run_data[1][2]
        target.append(t)
        results.append(r)
        count = count + 1
    return target, results

def evaluateCustomizedEsswithAllFunctions(representations, iids,ndim, budget = None, num_reps=1, storage_file=None):
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
    num_multiplications = len(iids)*num_reps
    arguments = list(product(representations, iids, range(num_reps)))

    for fid in Config.experiment_funcs:
        runFunction = partial(runCustomizedES, fid = fid, ndim=ndim, budget = budget)
        run_data = runParallelFunction(runFunction=runFunction,arguments=arguments)
        for rep in representations:
            reorganiseBBOBOutput(datapath+reprToString(rep)+'/',fid,ndim,iids,num_reps)

        targets, results = zip(*run_data)
        fitness_results = []

        for i, rep in enumerate(representations):

            # Preprocess/unpack results
            _,_, fitnesses, _ = (list(x) for x in zip(*results[i*num_multiplications:(i+1)*num_multiplications]))
            fitnesses = _trimListOfListsByLength(fitnesses)
            fitness = ESFitness(fitnesses)
            fitness_results.append(fitness)

            if not isinstance(rep, list):
                rep = rep.tolist()



def testEva():
    '''representations = [[0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 20, 0.25, 1, 1, 1, 1, 1, 1, 0.2, 0.955, 0.5, 0, 0.3, 0.5, 2],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 20, 0.25, 1, 1, 1, 1, 1, 1, 0.2, 0.955, 0.5, 0, 0.3, 0.5, 2],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 20, 0.25, 1, 1, 1, 1, 1, 1, 0.2, 0.955, 0.5, 0, 0.3, 0.5, 2],
                       [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 20, 0.25, 1, 1, 1, 1, 1, 1, 0.2, 0.955, 0.5, 0, 0.3, 0.5, 2],
                       [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 2, 20, 0.25, 1, 1, 1, 1, 1, 1, 0.2, 0.955, 0.5, 0, 0.3, 0.5, 2],
                       [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 2, 20, 0.25, 1, 1, 1, 1, 1, 1, 0.2, 0.955, 0.5, 0, 0.3, 0.5, 2],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 20, 0.25, 1, 1, 1, 1, 1, 1, 0.2, 0.955, 0.5, 0, 0.3, 0.5, 2]]'''

    #representations = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    representations = [[0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
                       [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 2]]
    for rep in representations:
        ensureFullLengthRepresentation(rep)
    #    , 1, 1, 1, 1, 0.2, 0.955, 0.5, 0, 0.3, 0.5, 2]]
    #import pdb
    #pdb.set_trace()

    print("I am HERE")
    #for i in Config.experiment_dims):
    for j in Config.experiment_funcs:
	if(j >= 13):
		results = evaluateCustomizedESs(representations, iids=range(Config.ES_num_runs), ndim=35, fid=j, budget=Config.GA_budget, num_reps=1, storage_file=None)


    #results = evaluateCustomizedESs(representations,fid=2, ndim=2, num_reps=1, budget= Config.GA_budget,iids=range(Config.ES_num_runs), storage_file="21")
    return results


def main():
    #result = _runES(1,10,1,100,Config.GA_budget)

    testEva()
    print("1\n")

if __name__ == '__main__':
        main()
