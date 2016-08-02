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
from collections import Counter
from code import getPrintName, getOpts
from code.Utils import ESFitness

np.set_printoptions(linewidth=156)
brute_location = 'C:\\Users\\Sander\\Dropbox\\Liacs\\DAS4\\Experiments\\BF runs'

ga_location = 'C:\\Users\\Sander\\Dropbox\\Liacs\\DAS4\\Experiments\\GA runs'  # laptop
# ga_location = '/home/sander/Dropbox/Liacs/Semester12/Thesis/test_results'  # desktop
dims = [2, 3, 5, 10, 20]
functions = range(1, 25)
np_save_names = ['time_spent', 'generation_sizes', 'sigma', 'best_result', 'best_fitness']


def reprToString(representation):
    max_length = 11  # Hardcoded
    return ''.join([str(i) for i in representation[:max_length]])


def reprToInt(representation):
    # Hardcoded
    max_length = 11
    factors = [2304, 1152, 576, 288, 144, 72, 36, 18, 9, 3, 1]
    result = 0
    for i in range(max_length):
        result += representation[i] * factors[i]

    return result


def getBestEs():
    os.chdir(ga_location)

    results = {dim: {} for dim in dims}

    for dim in dims:
        results[dim] = {func: {} for func in functions}

        # for file in final_files:
        for func in functions:
            filename = 'final_stats\\final_GA_results_{}dim_f{}.npz'.format(dim, func)
            x = np.load(filename)

            # for data in ['time_spent', 'best_result']:
            for data in x.files:  # ['time_spent', 'generation_sizes', 'sigma', 'best_result', 'best_fitness']
                results[dim][func][data] = x[data]

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
            print("  F{}:\t{}".format(func, results[dim][func]['best_result']))

    pprint.pprint(to_store)
    with open('ES_per_experiment.json', 'w') as json_out:
        json.dump(to_store, json_out)


def printTable(results):
    print('\\hline')
    print('F-ID & N & GA & FCE & ERT \\\\')
    print('\\hline')
    print('\\hline')
    for fid in functions:
        for dim in dims:
            result, fit = results[dim][fid]
            string = ''
            # for i in range(len(result)):
            for i in range(11):
                string += str(result[i])

            print('F{0} & {1} & {2} & {3:.3g} & {4}\\\\'.format(fid, dim, string, fit.FCE, fit.ERT))
        print('\\hline')


def printCompTable(bf, ga):
    print('\\hline')
    print('F-ID & N & Brute Force & FCE & ERT & GA & FCE & ERT \\\\')
    print('\\hline')
    print('\\hline')
    for fid in functions:
        for dim in dims:
            bf_result, bf_fit = bf[dim][fid]
            ga_result, ga_fit = ga[dim][fid]
            ga_diff = ''
            bf_string = ''
            for i in range(len(bf_result)):
                bf_string += str(bf_result[i])
                if bf_result[i] != ga_result[i]:
                    ga_diff += '\\underline{}{}{}'.format('{', ga_result[i], '}')
                else:
                    ga_diff += str(ga_result[i])

            bf_ert = np.inf if bf_fit.ERT is None else bf_fit.ERT
            ga_ert = np.inf if ga_fit.ERT is None else ga_fit.ERT
            print('F{0} & {1} & {2} & {3:.3g} & {4:.3g} & {5} & {6:.3g} & {7:.3g}\\\\'.format(
                fid, dim, bf_string, bf_fit.FCE, bf_ert, ga_diff, ga_fit.FCE, ga_ert
            ))
        print('\\hline')


def printGATable():

    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()
    ga_results = {dim: {fid: (results[dim][fid]['best_result'], min(results[dim][fid]['best_fitness'])) for fid in functions} for dim in dims}
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
    ga_results = {dim: {fid: (results[dim][fid]['best_result'], min(results[dim][fid]['best_fitness'])) for fid in functions} for dim in dims}

    os.chdir(brute_location)
    with open('brute_results.dat') as f:
        bf_results = cPickle.load(f)
    printCompTable(bf_results, ga_results)


def printIntCount(results):

    from collections import Counter

    all_strings = []
    for fid in functions:
        for dim in dims:
            all_strings.append(results[dim][fid][0])

    choice_lists = zip(*all_strings)
    counters = [Counter(int_list) for int_list in choice_lists]
    # print(counters)
    # print()
    for count in counters:
        for i in range(3):
            print(count[i], ' ', end='')
        print()


def printGAcount():
    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()
    ga_results = {dim: {fid: (results[dim][fid]['best_result'], min(results[dim][fid]['best_fitness'])) for fid in functions} for dim in dims}
    printIntCount(ga_results)


def printDoubleCount():

    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()
    ga_results = {dim: {fid: (results[dim][fid]['best_result'], min(results[dim][fid]['best_fitness'])) for fid in functions} for dim in dims}
    printIntCount(ga_results)
    print()
    os.chdir(brute_location)
    with open('brute_results.dat') as f:
        bf_results = cPickle.load(f)
    printIntCount(bf_results)


def createGARunPlots():
    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()

    # matplotlib.rcParams.update({'font.size': 18})
    plt.figure(figsize=(12,6.75))

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

            plt.subplot(1, 2, 1)
            plt.plot([x.FCE for x in best_found_ever], label='{}-dim'.format(dim))
            plt.subplot(1, 2, 2)
            plt.plot([x.ERT for x in best_found_ever], label='{}-dim'.format(dim))

        plt.suptitle("F{}".format(func))

        plt.subplot(1, 2, 1)
        plt.yscale('log')
        plt.xlabel('Generation')
        plt.ylabel('FCE')
        plt.legend(loc=0)

        plt.subplot(1, 2, 2)
        plt.yscale('log')
        plt.xlabel('Generation')
        plt.ylabel('ERT')
        plt.legend(loc=0)

        # plt.savefig('img/F{}_log.png'.format(func), bbox_inches='tight')
        plt.savefig('img/F{}_log.pdf'.format(func), bbox_inches='tight')


def findBestFromBF():
    os.chdir(brute_location)

    raw_fname = 'data\\bruteforce_{}_f{}.tdat'
    results = {dim: {} for dim in dims}

    for dim in dims:
        for fid in functions:
            best_es = None
            best_result = ESFitness()
            with open(raw_fname.format(dim, fid)) as f:
                for line in f:
                    parts = line.split('\t')
                    ES = eval(parts[0])
                    fitness = eval(parts[1])
                    if fitness < best_result:
                        best_result = fitness
                        best_es = ES
            results[dim][fid] = (best_es, best_result)

    with open('brute_results.dat', 'w') as f:
        cPickle.dump(results, f)


def checkFileSizesBF():
    os.chdir(brute_location)

    raw_fname = 'data\\bruteforce_{}_f{}.tdat'

    for dim in dims:
        print(dim)
        for fid in functions:
            with open(raw_fname.format(dim, fid)) as f:
                lines = [line for line in f]
                if len(lines) != 4608:
                    print("File bruteforce_{}_f{}.tdat does not contain 4608 entries! ({})".format(dim, fid, len(lines)))


def findGAInRankedBF():
    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    ga_results = x['results'].item()

    os.chdir(brute_location)
    raw_fname = 'data\\bruteforce_{}_f{}.tdat'
    results = {dim: {} for dim in dims}

    for dim in dims:
        for fid in functions:
            ga = reprToInt(ga_results[dim][fid]['best_result'])
            fit = ga_results[dim][fid]['best_fitness'][-1]

            bf_results = []
            with open(raw_fname.format(dim, fid)) as f:
                for line in f:
                    parts = line.split('\t')
                    ES = eval(parts[0])
                    fitness = eval(parts[1])
                    bf_results.append((reprToInt(ES), reprToString(ES), fitness))

            bf_results.sort(key=lambda a: a[2])
            indexes = [a[0] for a in bf_results]
            ga_index = indexes.index(ga)

            # Where does the GA-found ERT/FCE result rank in the brute-force results?
            fit_index = 0
            max_index = len(bf_results)
            while fit_index < max_index and fit > bf_results[fit_index][2]:
                fit_index += 1

            results[dim][fid] = (ga, fit_index, ga_index, indexes)
            print("{:>2}D F{:>2}:  GA {:>4} is ranked {:>4} ({:>4})\t\t\t GA: {} \t BF[0]: {}".format(dim, fid, ga,
                                                                                             fit_index, ga_index,
                                                                                             fit, bf_results[0][2]))

    with open('rank_ga_in_bf.dat', 'w') as f:
        cPickle.dump(results, f)

def printGAInRankedBF():

    os.chdir(brute_location)
    with open('rank_ga_in_bf.dat') as f:
        results = cPickle.load(f)

    fit_ranks = []
    ga_ranks = []
    for dim in dims:
        for fid in functions:
            fit_ranks.append(results[dim][fid][1])
            ga_ranks.append(results[dim][fid][2])

    count = Counter(fit_ranks)
    print(sorted(count.items(), key=lambda x: x[0]))

    count = Counter(ga_ranks)
    print(sorted(count.items(), key=lambda x: x[0]))

if __name__ == '__main__':

    ### GA STUFF ###
    # storeResults()
    # printResults()

    # createGARunPlots()
    # printGATable()
    # printGAcount()
    # storeRepresentation()


    # os.chdir(ga_location)
    # with open('ES_per_experiment.json') as infile:
    #     x = json.load(infile)
    # pprint.pprint(x)

    ### Brute Force STUFF ###

    # checkFileSizesBF()
    # findBestFromBF()
    # findGAInRankedBF()
    printGAInRankedBF()
    # printDoubleTable()
    # printDoubleCount()

    pass

