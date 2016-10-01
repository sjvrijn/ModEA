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
from collections import Counter, namedtuple
from datetime import timedelta
from code import getPrintName, getOpts
from code.Utils import ESFitness, intToRepr, reprToInt, reprToString

np.set_printoptions(linewidth=156)
brute_location = 'C:\\Users\\Sander\\Dropbox\\Liacs\\DAS4\\Experiments\\BF runs'
raw_bfname = 'data\\bruteforce_{}_f{}.tdat'

ga_location = 'C:\\Users\\Sander\\Dropbox\\Liacs\\DAS4\\Experiments\\GA runs'  # laptop
# ga_location = '/home/sander/Dropbox/Liacs/Semester12/Thesis/test_results'  # desktop
dimensions = [2, 3, 5, 10, 20]
functions = range(1, 25)
subgroups = [
    [1,  2,  3,  4,  5],
    [6,  7,  8,  9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24]
]
np_save_names = ['time_spent', 'generation_sizes', 'sigma', 'best_result', 'best_fitness']


### Utility functions ###
def BFFileToFitnesses(filename):
    """
        Given a brute_force filename, load all associated ESFitness objects and return them in tuples with both an
        numerical and string value of the relevant ES-structure. N.B.: the filename is **not** checked for correctness

        :param filename:    A string expected to be in the format: 'data\\bruteforce_{}_f{}.tdat'
        :return:            List of tuples (reprToInt(ES), reprToString(ES), ESFitness), sorted by ESFitness
    """

    os.chdir(brute_location)
    bf_results = []
    ES_and_result = namedtuple('ES_and_result', ['ES', 'fitness'])
    with open(filename) as f:
        for line in f:
            parts = line.split('\t')
            bf_results.append(ES_and_result(eval(parts[0]), eval(parts[1])))
    return bf_results


### GA STUFF ###
def getBestEs():
    os.chdir(ga_location)
    results = {dim: {} for dim in dimensions}
    for dim in dimensions:
        results[dim] = {func: {} for func in functions}
        # for file in final_files:
        for fid in functions:
            filename = 'final_stats\\final_GA_results_{}dim_f{}.npz'.format(dim, fid)
            x = np.load(filename)
            # for data in ['time_spent', 'best_result']:
            for data in x.files:  # ['time_spent', 'generation_sizes', 'sigma', 'best_result', 'best_fitness']
                results[dim][fid][data] = x[data]
    return results


def storeResults():
    results = getBestEs()
    np.savez('final_GA_results.npz', results=results, dims=dimensions, functions=functions, np_save_names=np_save_names)


def printResults():

    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()
    for dim in dimensions:
        print("{}-dimensional:".format(dim))
        for func in functions:
            print("  F{}:\t{} {}".format(func, results[dim][func]['best_result'], getPrintName(getOpts(results[dim][func]['best_result']))))


def createGARunPlots():
    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()

    matplotlib.rcParams.update({'font.size': 14})
    plt.figure(figsize=(8,4.5))

    for func in functions:
        print("F{}:".format(func))

        plt.clf()
        for dim in dimensions:
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

        plt.suptitle("Convergence for F{}".format(func), y=.99)

        plt.subplot(1, 2, 1)
        plt.yscale('log')
        plt.xlabel('Generation')
        plt.ylabel('FCE')
        plt.legend(loc=0, prop={'size':11}, labelspacing=0.15)

        plt.subplot(1, 2, 2)
        plt.yscale('log')
        plt.xlabel('Generation')
        plt.ylabel('ERT')
        plt.legend(loc=0, prop={'size':11}, labelspacing=0.15)

        plt.tight_layout()

        # plt.savefig('img/F{}_log.png'.format(func), bbox_inches='tight')
        plt.savefig('img/F{}_log.pdf'.format(func), bbox_inches='tight')


def printTable(results):
    print('\\hline')
    print('F-ID & N & GA & FCE & ERT \\\\')
    print('\\hline')
    print('\\hline')
    for fid in functions:
        for dim in dimensions:
            result, fit = results[dim][fid]
            string = ''
            # for i in range(len(result)):
            for i in range(11):
                string += str(result[i])

            print('F{0} & {1} & {2} & {3:.3g} & {4}\\\\'.format(fid, dim, string, fit.FCE, fit.ERT))
        print('\\hline')


def printGATable():

    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()
    ga_results = {dim: {fid: (results[dim][fid]['best_result'], min(results[dim][fid]['best_fitness'])) for fid in functions} for dim in dimensions}
    printTable(ga_results)


def printGAcount():
    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()
    ga_results = {dim: {fid: (results[dim][fid]['best_result'], min(results[dim][fid]['best_fitness'])) for fid in functions} for dim in dimensions}
    printIntCount(ga_results)


def storeRepresentation():

    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()

    to_store = {dim: {} for dim in dimensions}
    for dim in dimensions:
        print("{}-dimensional:".format(dim))
        for func in functions:
            to_store[dim][func] = results[dim][func]['best_result'].tolist()
            print("  F{}:\t{}".format(func, results[dim][func]['best_result']))

    pprint.pprint(to_store)
    with open('ES_per_experiment.json', 'w') as json_out:
        json.dump(to_store, json_out)


def GAtimeSpent():
    os.chdir(ga_location)
    times = []
    for dim in dimensions:
        for fid in functions:
            filename = 'final_stats\\final_GA_results_{}dim_f{}.npz'.format(dim, fid)
            x = np.load(filename)
            times.append(x['time_spent'])
    print(min(times), max(times), sum(times, timedelta(0)) // len(times))


def printCompTable(bf, ga):
    print('\\hline')
    print('F-ID & N & Brute Force & FCE & ERT & GA & FCE & ERT \\\\')
    print('\\hline')
    print('\\hline')
    for fid in functions:
        for dim in dimensions:
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


### BF STUFF ###
def printBFTable():

    os.chdir(brute_location)
    with open('brute_results.dat') as f:
        x = cPickle.load(f)
        printTable(x)


def printDoubleTable():

    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()
    ga_results = {dim: {fid: (results[dim][fid]['best_result'], min(results[dim][fid]['best_fitness'])) for fid in functions} for dim in dimensions}

    os.chdir(brute_location)
    with open('brute_results.dat') as f:
        bf_results = cPickle.load(f)
    printCompTable(bf_results, ga_results)


def printIntCount(results, fids=None, dims=None):

    if fids is None:
        fids = functions
    if dims is None:
        dims = dimensions

    from collections import Counter

    all_strings = []
    for fid in fids:
        for dim in dims:
            all_strings.append(results[dim][fid][0])

    choice_lists = zip(*all_strings)
    counters = [Counter(int_list) for int_list in choice_lists]
    # print(counters)
    # print()
    for count in counters[:11]:
        n = (count[0] + count[1] + count[2]) / 100
        print("{:>5.3} {:>5.3} {:>5.3}".format(count[0]/n, count[1]/n, count[2]/n))
        # for i in range(3):
        #     print(count[i], ' ', end='')
        # print()


def printDoubleCount(fids=None, dims=None):

    if fids is None:
        fids = functions
    if dims is None:
        dims = dimensions

    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()
    ga_results = {dim: {fid: (results[dim][fid]['best_result'], min(results[dim][fid]['best_fitness'])) for fid in fids} for dim in dims}
    printIntCount(ga_results, fids, dims)
    print()
    os.chdir(brute_location)
    with open('brute_results.dat') as f:
        bf_results = cPickle.load(f)
    printIntCount(bf_results, fids, dims)


def storeBestFromBF():

    results = {dim: {} for dim in dimensions}
    for dim in dimensions:
        for fid in functions:

            bf_results = BFFileToFitnesses(raw_bfname.format(dim, fid))
            bf_results.sort(key=lambda a: a.fitness)
            results[dim][fid] = bf_results[0]

    os.chdir(brute_location)
    with open('brute_results.dat', 'w') as f:
        cPickle.dump(results, f)


def checkFileSizesBF():
    os.chdir(brute_location)

    for dim in dimensions:
        print(dim)
        for fid in functions:
            with open(raw_bfname.format(dim, fid)) as f:
                lines = [line for line in f]
                if len(lines) != 4608:
                    print("File bruteforce_{}_f{}.tdat does not contain 4608 entries! ({})".format(dim, fid, len(lines)))


def findGAInRankedBF():
    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    ga_results = x['results'].item()

    results = {dim: {} for dim in dimensions}

    for dim in dimensions:
        for fid in functions:
            ga = reprToInt(ga_results[dim][fid]['best_result'])
            fit = ga_results[dim][fid]['best_fitness'][-1]

            bf_results = BFFileToFitnesses(raw_bfname.format(dim, fid))
            bf_results.sort(key=lambda a: a.fitness)
            indexes = [reprToInt(a.ES) for a in bf_results]
            ga_index = indexes.index(ga)

            # Where does the GA-found ERT/FCE result rank in the brute-force results?
            fit_index = 0
            max_index = len(bf_results)
            while fit_index < max_index and fit > bf_results[fit_index].fitness:
                fit_index += 1

            results[dim][fid] = (ga, fit_index, ga_index, indexes)
            print("{:>2}D F{:>2}:  GA {:>4} is ranked {:>4} ({:>4})\t\t\t GA: {} \t BF[0]: {}".format(dim, fid, ga,
                                                                                                      fit_index, ga_index,
                                                                                                      fit,
                                                                                                      bf_results[0].fitness))

    with open('rank_ga_in_bf.dat', 'w') as f:
        cPickle.dump(results, f)


def findGivenInRankedBF(dim, fid, given):

    bf_results = BFFileToFitnesses(raw_bfname.format(dim, fid))
    bf_results.sort(key=lambda a: a.fitness)
    indexes = [reprToInt(a.ES) for a in bf_results]

    results = []
    for ES in given:
        results.append((ES, indexes.index(ES), bf_results[indexes.index(ES)].fitness))
    return results


def getBestFromRankedBF(dim, fid, num=10):

    bf_results = BFFileToFitnesses(raw_bfname.format(dim, fid))
    bf_results.sort(key=lambda a: a.fitness)
    indexes = [reprToInt(a.ES) for a in bf_results]

    results = []
    for i in range(num):
        results.append((indexes[i], i, bf_results[i].fitness))

    return results


def printBestFromRankedBF():
    for dim in dimensions:
        for fid in functions:
            print("Results for F{} in {}dim:".format(fid, dim))
            # print(findGivenInRankedBF(dim, 1, given))
            results = getBestFromRankedBF(dim, fid, num=5)
            for ES, rank, fit in results:
                print("Rank: {0:>4}\t{1:>33}\t{2}".format(rank+1, intToRepr(ES), fit))
            print()


def printGAInRankedBF():

    os.chdir(brute_location)
    with open('rank_ga_in_bf.dat') as f:
        results = cPickle.load(f)

    fit_ranks = []
    ga_ranks = []
    for dim in dimensions:
        for fid in functions:
            fit_ranks.append(results[dim][fid][1])
            ga_ranks.append(results[dim][fid][2])

    count = Counter(fit_ranks)
    fit_ranking = sorted(count.items(), key=lambda x: x[0])
    fit_ranking.reverse()
    count = Counter(ga_ranks)
    struct_ranking = sorted(count.items(), key=lambda x: x[0])
    struct_ranking.reverse()

    f = s = 0
    full_ranking = []
    while len(fit_ranking) > 0 or len(struct_ranking) > 0:
        if len(fit_ranking) > 0:
            fit_rank, fit_count = fit_ranking[-1]
        else:
            fit_rank = fit_count = 1e5
        str_rank, str_count = struct_ranking[-1]

        if fit_rank == str_rank:
            full_ranking.append((fit_rank, fit_count, str_count))
            fit_ranking.pop()
            struct_ranking.pop()
        elif fit_rank < str_rank:
            full_ranking.append((fit_rank, fit_count, 0))
            fit_ranking.pop()
        elif str_rank < fit_rank:
            full_ranking.append((str_rank, 0, str_count))
            struct_ranking.pop()

    for rank, f_count, s_count in full_ranking:
        print("{:>4} & {:>2} & {:>2} \\\\".format(rank+1, f_count, s_count))


def printBFFitDistances():
    for dim in dimensions:
        for fid in functions:
            bf_results = BFFileToFitnesses(raw_bfname.format(dim, fid))
            bf_results.sort(key=lambda a: a.fitness)
            print("{:>2}dim F{:>2}: {}".format(dim, fid, [str(res.fitness) for res in bf_results[::100]]))


def correlationMatrix(fids=None):

    if fids is None:
        fids = functions

    os.chdir(ga_location)
    x = np.load('final_GA_results.npz')
    results = x['results'].item()
    ga_results = {dim: {fid: results[dim][fid]['best_result'] for fid in fids} for dim in dimensions}

    os.chdir(brute_location)
    with open('brute_results.dat') as f:
        bf_all = cPickle.load(f)
    bf_results = {dim: {fid: bf_all[dim][fid][0] for fid in fids} for dim in dimensions}

    ga_corr = np.zeros((11, 11))
    bf_corr = np.zeros((11, 11))

    for dim in dimensions:
        for fid in fids:
            ga = ga_results[dim][fid]
            bf = bf_results[dim][fid]

            for i in range(11):
                for j in range(i, 11):
                    ga_corr[i, j] += 1 if ga[i] * ga[j] != 0 else 0
                    bf_corr[i, j] += 1 if bf[i] * bf[j] != 0 else 0

    print(ga_corr)
    print(bf_corr)


def printComparisonBestAndGivenBF():

    # Default options: just select each module separately. Add/remove choices as you wish
    given = [
        reprToInt([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        reprToInt([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        reprToInt([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        reprToInt([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        reprToInt([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        reprToInt([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        reprToInt([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        reprToInt([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        reprToInt([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        reprToInt([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        reprToInt([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        reprToInt([0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]),
        reprToInt([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        reprToInt([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]),
    ]
    for dim in dimensions:
        for fid in functions:
            print("Results for F1 in {}dim:".format(dim))
            # print(findGivenInRankedBF(dim, 1, given))
            results = findGivenInRankedBF(dim, fid, given)
            for ES, rank, fit in results:
                print("{0:>33}\t{1:>4}\t{2}".format(intToRepr(ES), rank, fit))
            print()


if __name__ == '__main__':

    ### GA STUFF ###
    # storeResults()
    # printResults()

    # createGARunPlots()
    # printGATable()
    # printGAcount()
    # storeRepresentation()
    # GAtimeSpent()


    # os.chdir(ga_location)
    # with open('ES_per_experiment.json') as infile:
    #     x = json.load(infile)
    # pprint.pprint(x)

    ### Brute Force STUFF ###

    # checkFileSizesBF()
    # findBestFromBF()
    # printBFFitDistances()

    # findGAInRankedBF()
    # printGAInRankedBF()
    # printDoubleTable()
    # printDoubleCount()
    # correlationMatrix()

    # for i, subgroup in enumerate(subgroups):
    #     print(i)
    #     printDoubleCount(fids=subgroup)
    #     correlationMatrix(fids=subgroup)

    # for dim in dimensions:
    #     print(dim)
    #     printDoubleCount(dims=[dim])

    printComparisonBestAndGivenBF()
    printBestFromRankedBF()



    pass

