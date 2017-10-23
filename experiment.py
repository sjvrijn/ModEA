#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is written by F.Ye
# To do test


from codes import Config
from functools import partial
from EvolvingES import runCustomizedES,evaluateCustomizedESs

def _evaluateESSpace(iid,ndim,fid,rep,budget):
    storage_file = 'Result'
    representation = [0,0,0,0,0,0,0,0,0,0,0,20,0.25,1,1,1,1,1,1,0.2,0.955,0.5,0,0.3,0.5,2]
    #fitness_result = EvEs.evaluateCustomizedESs(representation,iids=range(Config.ES_num_runs),ndim=Config.experiment_dims,fid=1,budget=Config.GA_budget,num_reps=100,storage_file=storage_file)
    #run_data = runCustomizedES(representation=representation,iid=iid,ndim=ndim, fid=fid, rep =rep,budget=budget)
    run_data = _runES(representation=representation,iid=iid,ndim=ndim, fid=fid, rep =rep,budget=budget)
    target = run_data[0]
    results = run_data[1]
    rtarget = [False] * rep
    rtarget = [rtarget] * len(results)

    space = 0.0
    for j in rep:
        count = 0
        for i in len(results):
            if j == 0:
                if results[i][j] <= target[i]:
                    rtarget[i][j] = True
                    count = count + 1
                else:
                    rtarget[i][j] = False
            else:
                if rtarget[i][j-1] == True:
                    rtarget[i][j] = True
                    count = count + 1
                else:
                    if results[i][j] <= target[i]:
                        rtarget[i][j] = True
                        count = count + 1
                    else:
                        rtarget[i][j] = False
        space = space + (count / len(results))

    return space

def _runES(representation, iid, rep, ndim, fid, budget,runtimes=10):
    count = 0
    target = []
    results = []
    while count < runtimes:
        run_data = runCustomizedES(representation=representation,iid=iid,ndim=ndim,fid=fid,rep=rep,budget=budget)
        t = run_data[0]
        r = run_data[1][2]
        target.append(t)
        results.append(r)
        count = count + 1
    return target, results

def testEva():
    representations = [[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 20, 0.25, 1, 1, 1, 1, 1, 1, 0.2, 0.955, 0.5, 0, 0.3, 0.5, 2],
                      [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 20, 0.25, 1, 1, 1, 1, 1, 1, 0.2, 0.955, 0.5, 0, 0.3, 0.5, 2],
                      [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 20, 0.25, 1, 1, 1, 1, 1, 1, 0.2, 0.955, 0.5, 0, 0.3, 0.5, 2]]

    results = evaluateCustomizedESs(representations, iids=[1,2,3], ndim=Config.experiment_dims[1], fid=1, budget=Config.GA_budget, num_reps=1, storage_file=None)
    return results

def main():
    #result = _runES(1,10,1,100,Config.GA_budget)
    testEva()
    print("1\n")

if __name__ == '__main__':
        main()
