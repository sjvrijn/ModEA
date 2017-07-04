#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

### NOTE: Do not remove these 'unused' imports! ###
# The following are imports that are required by the functions that are passed to this MPI slave in order to run
from code.GA import _fetchResults, evaluate_ES  # Required for the MPI calls for the GA

import numpy as np
from code import MPI
comm = MPI.COMM_SELF.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()



def runSlaveRun():


    np.set_printoptions(linewidth=1000)
    function = None
    options = None

    # print("Process {}/{} reporting for duty!".format(rank, size))

    function = comm.bcast(function, root=0)
    arguments = comm.scatter(options, root=0)

    results = function(arguments)

    comm.Barrier()
    comm.gather(results, root=0)
    comm.Disconnect()


if __name__ == '__main__':

    runSlaveRun()
