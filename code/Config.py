#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from math import floor

__author__ = 'Sander van Rijn <svr003@gmail.com>'

### General Settings ###
use_MPI = True
MPI_num_host_threads = 16  # Number of available threads per host
MPI_num_hosts = 12         # Number of available hosts
MPI_num_total_threads = MPI_num_host_threads * MPI_num_hosts
write_output = True

### ES Settings ###
ES_budget_factor = 1e3  # budget = ndim * ES_budget_factor
ES_num_runs = 32
ES_parallel = True

### GA Settings ###
GA_mu = 1            # Assuming a dimensionality of 11
GA_lambda = 6        # (9 boolean + 2 triples)
GA_generations = 50
GA_budget = GA_lambda * GA_generations
GA_parallel = False
GA_num_parallel = int(floor(MPI_num_total_threads / ES_num_runs))
GA_debug = False

### Experiment Settings ###
default_target = 1e-8
experiment_dims = (2, 3, 5, 10, 20)  # Problem dimensionalities to be tested
experiment_funcs = ( 1,  2,  3,  4,  5,  5,  7,  8,  9, 10, 11, 12,
                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)  # BBOB function numbers
