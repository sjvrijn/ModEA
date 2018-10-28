#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

### NOTE: Do not remove these 'unused' imports! ###
# The following are imports that are required by the functions that are passed to this MPI slave in order to run

import numpy as np
from EvolvingES import MPI
comm = MPI.COMM_SELF.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()



def runSlaveRun():
    """
        This function has the sole purpose of performing the distributed computations in an MPI-setting, by running the
        broadcast function on the scattered arguments.

        N.B.: The broadcast function must be imported in this file for this method to work!

        To use this function, call as follows:
        >>> comm = MPI.COMM_SELF.Spawn(sys.executable, args=['MPI_slave.py'], maxprocs=num_procs)
        >>> comm.bcast(runFunction, root=MPI.ROOT)
        >>> comm.scatter(arguments, root=MPI.ROOT)
        >>> comm.Barrier()
        >>> results = comm.gather(results, root=MPI.ROOT)
        >>> comm.Disconnect()
    """

    np.set_printoptions(linewidth=1000)
    function = None
    options = None

    # print("Process {}/{} reporting for duty!".format(rank, size))

    function = comm.bcast(function, root=0)
    arguments = comm.scatter(options, root=0)

    results = function(*arguments)

    comm.Barrier()
    comm.gather(results, root=0)
    comm.Disconnect()


if __name__ == '__main__':
    runSlaveRun()
