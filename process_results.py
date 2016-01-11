#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import numpy as np
import os
from code import getPrintName, getOpts


location = 'D:\\test_results'  # laptop
dims = [2, 3, 5, 10, 20]
functions = [3, 4, 7, 9, 10, 12, 13, 16, 17, 19, 20, 21, 23, 24]
np_save_names = ['time_spent', 'generation_sizes', 'sigma', 'best_result', 'best_fitness']


def getBestEs():
    os.chdir(location)

    results = {dim: {} for dim in dims}

    for dim in dims:
        folder = 'results_{}dim'.format(dim)
        os.chdir(folder)

        results[dim] = {func: {} for func in functions}

        # for file in final_files:
        for func in functions:
            filename = 'final_GA_results_{}dim_f{}.npz'.format(dim, func)
            x = np.load(filename)

            # for data in ['time_spent', 'best_result']:
            for data in x.files:  # ['time_spent', 'generation_sizes', 'sigma', 'best_result', 'best_fitness']
                results[dim][func][data] = x[data]

        os.chdir('..')

    return results


def storeResults():
    results = getBestEs()
    np.savez('final_GA_results.npz', results=results, dims=dims, functions=functions, np_save_names=np_save_names)


def printResults():

    os.chdir(location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()
    for dim in dims:
        print("{}-dimensional:".format(dim))
        for func in functions:
            # print(results[dim][func])
            print("  F{}:\t{} {}".format(func, results[dim][func]['best_result'], getPrintName(getOpts(results[dim][func]['best_result']))))

if __name__ == '__main__':

    # storeResults()
    # printResults()


    pass
