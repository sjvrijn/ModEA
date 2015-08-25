#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runs the timing experiment for PURE_RANDOM_SEARCH.

CAPITALIZATION indicates code adaptations to be made
This script as well as files bbobbenchmarks.py and fgeneric.py need to be
in the current working directory.

"""

import time
import numpy as np
import fgeneric
import bbobbenchmarks as bn

MAX_FUN_EVALS = '1e3 + 1e6/dim' # per run, adjust to default but prevent very long runs (>>30s)

def run_optimizer(fun, dim, maxfunevals, ftarget=-np.Inf):
    """start the optimizer, allowing for some preparation. 
    This implementation is an empty template to be filled 
    
    """
    # prepare
    x_start = 8. * np.random.rand(dim) - 4
    
    # call
    PURE_RANDOM_SEARCH(fun, x_start, maxfunevals, ftarget)

def PURE_RANDOM_SEARCH(fun, x, maxfunevals, ftarget):
    """samples new points uniformly randomly in [-5,5]^dim and evaluates
    them on fun until maxfunevals or ftarget is reached, or until
    1e8 * dim function evaluations are conducted.

    """
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    fbest = np.inf

    for iter in range(0, int(np.ceil(maxfunevals / popsize))):
        xpop = 10. * np.random.rand(popsize, dim) - 5.
        fvalues = fun(xpop)
        idx = np.argsort(fvalues)
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved 
            break

    return xbest

timings = []
runs = []
dims = []
for dim in (2, 3, 5, 10, 20, 40, 80, 160):
    nbrun = 0
    f = fgeneric.LoggingFunction('tmp').setfun(*bn.instantiate(8, 1))
    t0 = time.time()
    while time.time() - t0 < 30: # at least 30 seconds
        run_optimizer(f.evalfun, dim, eval(MAX_FUN_EVALS), f.ftarget)  # adjust maxfunevals
        nbrun = nbrun + 1
    timings.append((time.time() - t0) / f.evaluations)
    dims.append(dim)    # not really needed
    runs.append(nbrun)  # not really needed
    f.finalizerun()
    print '\nDimensions:',
    for i in dims:
        print ' %11d ' % i,
    print '\n      runs:',
    for i in runs:
        print ' %11d ' % i,
    print '\n times [s]:',
    for i in timings:
        print ' %11.1e ' % i, 
    print ''

