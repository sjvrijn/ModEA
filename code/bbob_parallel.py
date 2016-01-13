#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
"""
Created on Tue Feb 19 14:20:24 2013

@author: wangronin
"""


from itertools import product
import json
import os
import numpy as np
from time import time
from multiprocessing import Process
from numpy import array, mod
from functools import partial

from code import Config, getOpts, getPrintName
from code.Algorithms import customizedES
from bbob import fgeneric
import bbob.bbobbenchmarks as bn



def parallel_bbob(dim, fID, budget, opts, datapath, bbob_opts):
    """
    Parallel bbob experiment wrapper for ES algorithm test
    """
    
    # Set different seed for different processes
    T_0 = time()
    np.random.seed(mod(int(T_0)+os.getpid(), 1000))
    
    f = fgeneric.LoggingFunction(datapath, **bbob_opts)
    
    # small dimensions first, for CPU reasons
    for iinstance in range(1, 16):
        f.setfun(*bn.instantiate(fID, iinstance=iinstance))

        customizedES(dim, f.evalfun, budget, opts=opts)

        f.finalizerun()
            
    with open('./flag_stepsize_acc', 'a') as out:
        out.write('function ' + str(fID) + ' is done.\n')




if __name__ == '__main__':

    combos = list(product(Config.experiment_dims, Config.experiment_funcs))
    with open('~/src/master-thesis/ES_per_experiment.json') as filename:
        bests = json.load(filename)

    if Config.use_MPI:

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        dim, func = combos[rank]
        budget = Config.ES_budget_factor * dim
        bitstring = bests[str(dim)][str(func)]
        opts = getOpts(bitstring)
        datapath = '/var/scratch/sjvrijn/{}/{}/'.format(dim, func)
        bbob_opts = {
            'algid': 'Testing the {}'.format(getPrintName(opts)),
            'comments': 'evaluation budget setting: 1e4 * dim',
            'inputformat': 'col'
        }

        parallel_bbob(dim, func, budget, opts, datapath, bbob_opts)

    else:

        # procs = [Process(target=parallel_bbob, args=[DIM, OPTIONS, i]) for i in bn.nfreeIDs]
        #
        # # Starting the parallel computation
        # for p in procs: p.start()
        # for p in procs: p.join()

        raise NotImplementedError("Sorry Future person, Past me was too lazy to fix this. SvR -- 13-01-2016")
