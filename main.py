#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import sys
from bbob import bbobbenchmarks
from bbob import fgeneric
from code import getOpts, options
from code.Algorithms import CMA_ES, customizedES
from code.Algorithms import GA, MIES
from code.local import datapath

# Sets of noise-free and noisy benchmarks
free_function_ids = bbobbenchmarks.nfreeIDs
noisy_function_ids = bbobbenchmarks.noisyIDs


opts = {'algid': None,
        'comments': '<comments>',
        'inputformat': 'col'}  # 'row' or 'col'

def sysPrint(string):
    """ Small function to take care of the 'overhead' of sys.stdout.write + flush """
    sys.stdout.write(string)
    sys.stdout.flush()


def main():
    np.set_printoptions(linewidth=200)


if __name__ == '__main__':
    main()
