#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

### General Settings ###
use_MPI = True
write_output = True

### GA Settings ###
GA_mu = 1            # Assuming a dimensionality of 11
GA_lambda = 10       # (9 boolean + 2 triples)
GA_generations = 50
GA_budget = GA_lambda * GA_generations
GA_parallel = False
GA_debug = False

### ES Settings ###
ES_budget_factor = 1e3  # budget = ndim * ES_budget_factor
ES_num_runs = 32
ES_parallel = True

### Experiment Settings ###
default_target = 1e-8
experiment_dims = (2, 3, 5, 10, 20)  # Problem dimensionalities to be tested
experiment_funcs = ( 1,  2,  3,  4,  5,  5,  7,  8,  9, 10, 11, 12,
                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)  # BBOB function numbers
