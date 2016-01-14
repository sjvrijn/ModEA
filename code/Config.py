#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

### General Settings ###
use_MPI = True


### GA Settings ###
GA_mu = 1        # Assuming a dimensionality of 11
GA_lambda = 12   # (8 boolean + 3 triples)
GA_budget = 250  # Roughly 20 generations
GA_parallel = True
GA_debug = True

### ES Settings ###
ES_budget_factor = 1e3
ES_num_runs = 15
ES_parallel = True

### Experiment Settings ###
experiment_dims = (2, 3, 5, 10, 20)  # Problem dimensionalities to be tested
experiment_funcs = (3, 4, 7, 9, 10, 12, 13, 16, 17, 19, 20, 21, 23, 24)  # BBOB function numbers
