#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = 'Sander van Rijn <svr003@gmail.com>'

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import cPickle
from code import getPrintName, getOpts

np.set_printoptions(linewidth=156)
brute_location = 'C:\\Users\\Sander\\Dropbox\\Liacs\\Semester12\\Thesis\\brute_force\\final_results'

ga_location = 'D:\\test_results'  # laptop
# ga_location = '/home/sander/Dropbox/Liacs/Semester12/Thesis/test_results'  # desktop
dims = [2, 3, 5, 10, 20]
functions = [3, 4, 7, 9, 10, 12, 13, 16, 17, 19, 20, 21, 23, 24]
np_save_names = ['time_spent', 'generation_sizes', 'sigma', 'best_result', 'best_fitness']


def getBestEs():
    os.chdir(ga_location)

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

    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()
    for dim in dims:
        print("{}-dimensional:".format(dim))
        for func in functions:
            print("  F{}:\t{} {}".format(func, results[dim][func]['best_result'], getPrintName(getOpts(results[dim][func]['best_result']))))


def storeRepresentation():

    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()

    to_store = {dim: {} for dim in dims}
    for dim in dims:
        print("{}-dimensional:".format(dim))
        for func in functions:
            to_store[dim][func] = results[dim][func]['best_result'].tolist()
            # print("  F{}:\t{} {}".format(func, results[dim][func]['best_result']))

    pprint.pprint(to_store)
    with open('ES_per_experiment.json', 'w') as json_out:
        json.dump(to_store, json_out)


def printTable(results):
    print('\\hline')
    print('F-ID & Dim & Best ES Found\\\\')
    print('\\hline')
    print('\\hline')
    for fid in functions:
        for dim in dims:
            print('F{} & {} & {}\\\\'.format(fid, dim, getPrintName(getOpts(results[dim][fid]))))
        print('\\hline')


def printCompTable(bf, ga):
    print('\\hline')
    print('F-ID & Dim & Best ES & Best ES Found\\\\')
    print('\\hline')
    print('\\hline')
    for fid in functions:
        for dim in dims:
            bf_result = bf[dim][fid]
            ga_result = ga[dim][fid]
            ga_diff = ''
            bf_string = ''
            for i in range(len(bf_result)):
                bf_string += str(bf_result[i])
                if bf_result[i] != ga_result[i]:
                    ga_diff += '\\underline{}{}{}'.format('{', ga_result[i], '}')
                else:
                    ga_diff += str(ga_result[i])

            print('F{} & {} & {} & {}\\\\'.format(fid, dim, bf_string, ga_diff))
        print('\\hline')


def printGATable():

    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()
    ga_results = {dim: {fid: results[dim][fid]['best_result'] for fid in functions} for dim in dims}
    printTable(ga_results)


def printBFTable():

    os.chdir(brute_location)
    with open('brute_results.dat') as f:
        x = cPickle.load(f)
        printTable(x)


def printDoubleTable():

    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()
    ga_results = {dim: {fid: results[dim][fid]['best_result'] for fid in functions} for dim in dims}

    os.chdir(brute_location)
    with open('brute_results.dat') as f:
        bf_results = cPickle.load(f)
    printCompTable(bf_results, ga_results)


def createGARunPlots():
    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()

    matplotlib.rcParams.update({'font.size': 18})

    for func in functions:
        print("F{}:".format(func))

        plt.clf()
        for dim in dims:
            best_per_generation = results[dim][func]['best_fitness'][::12]
            best_found_ever = []
            for i, fit in enumerate(best_per_generation):
                if fit <= min(best_per_generation[:i+1]):
                    best_found_ever.append(fit)
                else:
                    best_found_ever.append(best_found_ever[i-1])

            plt.plot(best_found_ever, label='{}-dim'.format(dim))

        plt.title("F{}".format(func))
        plt.xlabel('Generation')
        plt.ylabel('Distance to Target')
        plt.legend(loc=0)

        plt.savefig('img/F{}.png'.format(func), bbox_inches='tight')
        plt.savefig('img/F{}.pdf'.format(func), bbox_inches='tight')


def findBestFromBF():
    os.chdir(brute_location)

    raw_fname = 'bruteforce_{}_f{}.tdat'
    results = {dim: {} for dim in dims}

    for dim in dims:
        for fid in functions:
            best_es = None
            best_result = np.inf
            with open(raw_fname.format(dim, fid)) as f:
                for line in f:
                    parts = line.split('\t')
                    ES = eval(parts[0])
                    fitness = np.median(eval(parts[1]))
                    if fitness < best_result:
                        best_result = fitness
                        best_es = ES
            results[dim][fid] = best_es

    with open('brute_results.dat', 'w') as f:
        cPickle.dump(results, f)


if __name__ == '__main__':

    ### GA STUFF ###
    # storeResults()
    # printResults()

    # createGARunPlots()
    # printTable()
    # storeRepresentation()


    # os.chdir(location)
    # with open('ES_per_experiment.json') as infile:
    #     x = json.load(infile)
    # pprint.pprint(x)

    ### Brute Force STUFF ###

    # findBestFromBF()
    printDoubleTable()

    pass

