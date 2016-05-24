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


# Hardcoded because mpi & file transfer are too much trouble to figure out 48 hours before the deadline
bests = {2: {3: [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
             4: [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 2],
             7: [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 2],
             9: [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
             10: [0, 0, 1, 1, 0, 1, 0, 0, 2, 0, 0],
             12: [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 2],
             13: [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
             16: [0, 0, 1, 1, 0, 0, 0, 1, 2, 0, 0],
             17: [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
             19: [0, 1, 0, 1, 1, 0, 0, 1, 2, 0, 0],
             20: [1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 2],
             21: [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
             23: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
             24: [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]},
         3: {3: [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2],
             4: [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
             7: [0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 2],
             9: [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
             10: [0, 0, 1, 1, 0, 1, 0, 0, 2, 0, 2],
             12: [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
             13: [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
             16: [0, 0, 1, 1, 0, 0, 0, 1, 2, 0, 2],
             17: [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
             19: [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 2],
             20: [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
             21: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2],
             23: [0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 2],
             24: [0, 0, 0, 1, 1, 0, 0, 0, 2, 0, 2]},
         5: {3: [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 2],
             4: [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 2],
             7: [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
             9: [1, 0, 0, 0, 0, 1, 0, 1, 2, 0, 1],
             10: [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
             12: [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
             13: [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
             16: [0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0],
             17: [0, 0, 1, 1, 0, 0, 0, 1, 2, 0, 0],
             19: [0, 1, 1, 1, 0, 0, 0, 0, 2, 0, 0],
             20: [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
             21: [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
             23: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2],
             24: [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 2]},
         10: {3: [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1],
              4: [0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 2],
              7: [0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 2],
              9: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2],
              10: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2],
              12: [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 2],
              13: [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
              16: [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2],
              17: [0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0],
              19: [0, 1, 1, 1, 0, 0, 0, 0, 2, 0, 2],
              20: [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 2],
              21: [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
              23: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 2],
              24: [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]},
         20: {3: [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
              4: [0, 1, 0, 1, 0, 0, 0, 0, 0, 2, 0],
              7: [0, 1, 0, 1, 0, 0, 1, 0, 2, 0, 0],
              9: [1, 0, 1, 1, 1, 1, 0, 0, 2, 0, 1],
              10: [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
              12: [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
              13: [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 2],
              16: [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 2],
              17: [0, 0, 1, 1, 0, 0, 0, 1, 2, 0, 2],
              19: [0, 1, 0, 0, 1, 0, 0, 0, 2, 0, 2],
              20: [1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
              21: [1, 1, 0, 1, 1, 1, 1, 1, 2, 0, 0],
              23: [0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0],
              24: [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0]}
         }


def parallel_bbob(dim, fID, budget, opts, datapath, bbob_opts, values=None):
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

        customizedES(dim, f.evalfun, budget, opts=opts, values=values)

        f.finalizerun()

    with open('./flag_stepsize_acc', 'a') as out:
        out.write('function ' + str(fID) + ' is done.\n')




if __name__ == '__main__':

    combos = list(product(Config.experiment_dims, Config.experiment_funcs))
    # with open('~/src/master-thesis/ES_per_experiment.json') as filename:
    #     bests = json.load(filename)

    if Config.use_MPI:

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        dim, func = combos[rank]
        budget = Config.ES_budget_factor * dim
        bitstring = bests[dim][func]
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
